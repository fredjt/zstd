/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */

/**
 * @file zstd_cuda_matchfinder.cu
 * @brief CUDA implementation of GPU-accelerated match finding for zstd.
 *
 * G6 design (see doc/gpu_matchfinding_design.md):
 *
 * The CPU ZSTD_fast match finder keeps a hash table (4-byte fingerprint ->
 * latest position) and, for each position, walks the per-hash chain of
 * previous positions, comparing candidates until the attempt budget
 * (2^searchLog) is used, and keeps the longest match found in chain order.
 * The walk is inherently sequential in POSITION, which is what a GPU wants
 * to parallelize.
 *
 * GPU pipeline per 128 KB block (block-local : the POC does not carry the
 * sliding window across blocks, so each block is matched against itself):
 *
 *  1. fillKeys   : one thread per position writes the composite sort key
 *                  key = (hash << 17) | position.  The block is at most
 *                  2^17 bytes, so the low 17 bits are the position and keys
 *                  are unique.
 *  2. radixSort  : 3 passes of a stable counting sort on 16-bit digits
 *                  (per-block shared histogram, a 256-thread scan over the
 *                  per-block histograms, per-lane in-digit rank + scatter).
 *                  Each pass is a handful of independent global accesses
 *                  per element, with no dependent load chains (the previous
 *                  insertion-sort + merge-tree design was latency bound).
 *                  The array ends up sorted by (hash, position).
 *  3. recordSlot : slot[position] = index of the position in the sorted
 *                  array (the element just before it in the array is the
 *                  closest previous same-hash position).
 *  4. rowBounds  : for every hash, the start and end indices of its sorted
 *                  row (a CSR layout : the row is the same-hash positions in
 *                  ascending order). Written by distinct threads of the row
 *                  with no cross-thread read, so there is no intra-kernel
 *                  race.
 *  5. searchMatches : one thread per position walks its row backwards from
 *                  slot[p]-1 (i.e. same-hash positions strictly before p,
 *                  closest first - exactly the CPU chain order, without the
 *                  table's aliasing), 8-byte-checks, counts (capped at 64 KB)
 *                  and keeps the longest candidate within the 2^searchLog
 *                  budget and the compression window.
 *  6. A D2H copy of the per-position candidates (8 bytes each, at most 1 MB)
 *     plus a sequential greedy scan on the host resolves overlaps into the
 *     final non-overlapping sequence list (device-side sequential passes
 *     are latency-bound : one dependent global load per position).
 *
 * The CPU's chain table aliases when 2^chainLog < block size, so the GPU
 * rows are a superset-clean view of the same information : same candidate
 * order for a large-enough chain table, which is what the ratio target
 * ("within ~5% of CPU greedy") requires, not bit-identical sequences.
 */

#include "zstd_cuda_matchfinder.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/* GPU error codes are the ZSTD_error_GPU_* entries of zstd_errors.h
 * (added in G1); ERROR(GPU_xxx) below expands to the matching value. */

/* ======================================================================== */
/* Device helpers                                                            */
/* ======================================================================== */

#define ZSTD_GPU_POS_BITS 17
#define ZSTD_GPU_POS_MASK ((1u << ZSTD_GPU_POS_BITS) - 1)

namespace
{
__device__ __forceinline__
unsigned ZSTD_gpu_read32(void const* ptr)
{
    unsigned val;
    memcpy(&val, ptr, 4);
    return val;
}

/**
 * Hash 4 bytes with the same multiplicative hash the CPU fast strategy uses
 * (ZSTD_hashPtr / 5-3 trick : top hashLog bits of the product).
 */
__device__ __forceinline__
unsigned ZSTD_gpu_hashPtr(void const* ptr, const unsigned hashLog)
{
    unsigned const h = ZSTD_gpu_read32(ptr);
    return (h * 2654435761U) >> (32 - hashLog);
}

/**
 * Composite sort key : (hash << 17) | position. Unique per position.
 */
__device__ __forceinline__
unsigned long long ZSTD_gpu_makeKey(const unsigned h, const unsigned p)
{
    return ((unsigned long long) h << ZSTD_GPU_POS_BITS) | (unsigned long long) p;
}

/**
 * Maximum match length the GPU counts. The per-position count loop is the
 * most expensive part of the search on repetitive content (every position
 * would count a ~128 KB match), so the GPU counts only up to this cap and
 * sets matchLength = cap when the match runs past it. The host greedy scan
 * (which has the source buffer) extends such truncated matches to their full
 * length, so the emitted sequences keep the exact CPU match lengths and the
 * ratio is unaffected.
 */
#define ZSTD_GPU_MATCH_CAP 64

/**
 * Count matching bytes between two pointers (8-byte chunks, then 4-byte,
 * then byte tail), capped at srcSize and cap.
 */
__device__ __forceinline__
size_t ZSTD_gpu_count(void const* ip, void const* match, void const* iend, size_t const cap)
{
    size_t srcSize = static_cast<size_t>(static_cast<unsigned char const*>(iend) - static_cast<unsigned char const*>(ip));
    if (srcSize > cap) srcSize = cap;
    size_t length = 0;
    while (length + 8 <= srcSize)
    {
        unsigned long long val1, val2;
        memcpy(&val1, static_cast<unsigned char const*>(ip) + length, 8);
        memcpy(&val2, static_cast<unsigned char const*>(match) + length, 8);
        if (const unsigned long long diff = val1 ^ val2)
        {
            length += (__ffsll(diff) - 1) / 8;
            return length;
        }
        length += 8;
    }
    while (length + 4 <= srcSize)
    {
        unsigned val1, val2;
        memcpy(&val1, static_cast<unsigned char const*>(ip) + length, 4);
        memcpy(&val2, static_cast<unsigned char const*>(match) + length, 4);
        if (const unsigned diff = val1 ^ val2)
        {
            length += (__ffs(diff) - 1) / 8;
            return length;
        }
        length += 4;
    }
    while (length < srcSize && static_cast<unsigned char const*>(ip)[length] == static_cast<unsigned char const*>(match)[length])
        length++;
    return length;
}
}

/**
 * Per-position best candidate found by the parallel search kernel.
 * offset == 0 means "no match at this position".
 */
struct ZSTD_GpuPosCandidate {
    unsigned offset;
    unsigned matchLength;
};

/* ======================================================================== */
/* Kernels                                                                   */
/* ======================================================================== */

/**
 * Fill the composite sort keys. Pads up to n2 (a multiple of the radix
 * block size) with the maximum 64-bit value so the padding sorts after
 * every real key and forms a single dummy row at the end of the array
 * that the search never references.
 */
__global__
static void ZSTD_gpu_fillKeys(unsigned char const* d_window, unsigned long long* d_keys,
                              const unsigned hashLog, const unsigned nValid, const unsigned n2)
{
    unsigned const p = (unsigned)(blockIdx.x * (size_t)blockDim.x + threadIdx.x);
    if (p >= n2) return;
    if (p < nValid)
        d_keys[p] = ZSTD_gpu_makeKey(ZSTD_gpu_hashPtr(d_window + p, hashLog), p);
    else
        d_keys[p] = ~0ULL;
}

/* The keys are 64-bit but real keys fit in 47 bits (hashLog <= 30, 17
 * position bits). The fill order is ascending position, i.e. the input is
 * already strictly sorted on the low 17 bits, so the first 8-bit digit
 * (bits 0-7) and, together with bit 16 carried by the input order, bits
 * 0-15 need no sorting pass : 4 passes over bits 16-47 finish the job.
 * 8-bit digits keep the per-block shared histogram at 1 KB. */
#define ZSTD_GPU_RADIX_BITS 8
#define ZSTD_GPU_RADIX_BINS (1u << ZSTD_GPU_RADIX_BITS)
#define ZSTD_GPU_RADIX_PASSES 4
#define ZSTD_GPU_BLOCK_THREADS 256

/**
 * Radix pass 1 : per-block histogram of the pass digit, in shared memory;
 * each block then stores its 256 counts into its own row of the global
 * per-block histogram (no inter-block contention : cell [b][d] is written
 * by block b only).
 */
__global__
static void ZSTD_gpu_radixHist(unsigned long long const* __restrict__ d_in, unsigned* __restrict__ d_blockHist,
                               const unsigned shift, const unsigned n)
{
    __shared__ unsigned s_hist[ZSTD_GPU_RADIX_BINS];
    if (threadIdx.x < ZSTD_GPU_RADIX_BINS) s_hist[threadIdx.x] = 0u;
    __syncthreads();
    unsigned const t = (unsigned)(blockIdx.x * (size_t)blockDim.x + threadIdx.x);
    if (t < n)
    {
        unsigned const d = (unsigned)((d_in[t] >> shift) & (ZSTD_GPU_RADIX_BINS - 1u));
        atomicAdd(&s_hist[d], 1u);
    }
    __syncthreads();
    if (threadIdx.x < ZSTD_GPU_RADIX_BINS)
        d_blockHist[blockIdx.x * ZSTD_GPU_RADIX_BINS + threadIdx.x] = s_hist[threadIdx.x];
}

/**
 * Radix pass 2 : for each digit, the exclusive prefix over the per-block
 * counts. Turns d_blockHist in place into d_blockOffset : where in the
 * per-digit region digit d of block b starts. Also computes d_digitStart :
 * the global output index where each digit's region begins (exclusive
 * prefix over the per-digit totals). One block of 256 threads (one per
 * digit) walks at most nBlocks entries each.
 */
__global__
static void ZSTD_gpu_radixScan(unsigned* __restrict__ d_blockHist, unsigned* __restrict__ d_digitStart,
                               const unsigned nBlocks)
{
    __shared__ unsigned s_total[ZSTD_GPU_RADIX_BINS];
    unsigned const d = threadIdx.x;
    unsigned run = 0u;
    for (unsigned b = 0; b < nBlocks; b++)
    {
        unsigned const v = d_blockHist[b * ZSTD_GPU_RADIX_BINS + d];
        d_blockHist[b * ZSTD_GPU_RADIX_BINS + d] = run;
        run += v;
    }
    s_total[d] = run; /* per-digit total across all blocks */
    __syncthreads();
    if (threadIdx.x == 0)
    {
        unsigned acc = 0u;
        for (unsigned i = 0; i < ZSTD_GPU_RADIX_BINS; i++)
        {
            unsigned const v = s_total[i];
            s_total[i] = acc;
            acc += v;
        }
    }
    __syncthreads();
    d_digitStart[d] = s_total[d];
}

/**
 * Radix pass 3 : stable scatter. Each lane's output index is
 * blockOffset[block][digit] + (number of earlier lanes with the same digit),
 * so elements with equal digits keep their input order (LSD radix needs
 * stability between passes). Lanes outside n use a sentinel digit that no
 * real digit can equal, so they never pollute the per-digit rank.
 */
__global__
static void ZSTD_gpu_radixScatter(unsigned long long const* __restrict__ d_in, unsigned long long* __restrict__ d_out,
                                  unsigned const* __restrict__ d_blockOffset, unsigned const* __restrict__ d_digitStart,
                                  const unsigned shift, const unsigned n)
{
    __shared__ unsigned s_digit[ZSTD_GPU_BLOCK_THREADS];
    __shared__ unsigned s_hist[ZSTD_GPU_RADIX_BINS];
    unsigned const t = (unsigned)(blockIdx.x * (size_t)blockDim.x + threadIdx.x);
    unsigned const d = (t < n) ? (unsigned)((d_in[t] >> shift) & (ZSTD_GPU_RADIX_BINS - 1u)) : 0xFFFFFFFFu;
    s_digit[threadIdx.x] = d;
    if (threadIdx.x < ZSTD_GPU_RADIX_BINS) s_hist[threadIdx.x] = 0u;
    __syncthreads();
    if (t < n) atomicAdd(&s_hist[d], 1u);
    __syncthreads();
    if (t < n)
    {
        /* rank within the block for this digit : the number of lanes at or
         * before this lane holding the same digit (includes this lane) */
        unsigned count = 0u;
        for (unsigned i = 0; i <= threadIdx.x; i++)
            if (s_digit[i] == d) count++;
        d_out[d_digitStart[d] + d_blockOffset[blockIdx.x * ZSTD_GPU_RADIX_BINS + d] + count - 1u] = d_in[t];
    }
}

/**
 * slot[position] = index of the position in the finally sorted array.
 */
__global__
static void ZSTD_gpu_recordSlot(unsigned long long const* __restrict__ d_sorted, unsigned* d_slot, const unsigned n2)
{
    unsigned const j = (unsigned)(blockIdx.x * (size_t)blockDim.x + threadIdx.x);
    if (j >= n2) return;
    unsigned const p = (unsigned)(d_sorted[j] & ZSTD_GPU_POS_MASK);
    d_slot[p] = j;
}

/**
 * CSR row boundaries : for every real hash, the start and end indices of its
 * sorted row in the final array. Each of the two boundary threads of a row
 * writes to a different array (rowStart / rowEnd), so there is no
 * intra-kernel data race.
 *
 * The padding keys (~0ULL) sort last. When hashLog > 17 the padding hash
 * ((~0ULL) >> 17) falls inside the real hash space, so padding is detected
 * by exact key match, not by hash range : a row whose last element is
 * directly followed by padding is still terminated correctly.
 */
__global__
static void ZSTD_gpu_rowBounds(unsigned long long const* __restrict__ d_sorted, unsigned* d_rowStart, unsigned* d_rowEnd,
                               const unsigned hashLog, const unsigned n2)
{
    unsigned const j = (unsigned)(blockIdx.x * (size_t)blockDim.x + threadIdx.x);
    if (j >= n2) return;
    unsigned long long const key = d_sorted[j];
    if (key == ~0ULL) return; /* padding : the real hashes always fit in hashLog bits */
    unsigned const h = (unsigned)(key >> ZSTD_GPU_POS_BITS);
    unsigned long long const prev = (j > 0) ? d_sorted[j - 1] : ~0ULL;
    unsigned long long const next = (j + 1 < n2) ? d_sorted[j + 1] : ~0ULL;
    if (prev == ~0ULL || (unsigned)(prev >> ZSTD_GPU_POS_BITS) != h)
        d_rowStart[h] = j;
    if (next == ~0ULL || (unsigned)(next >> ZSTD_GPU_POS_BITS) != h)
        d_rowEnd[h] = j;
    (void) hashLog;
}

/**
 * Parallel search : one thread per position. Walks its hash row backwards
 * from the closest previous same-hash position (CSR layout : strictly
 * decreasing positions, no aliasing, no sentinel walk), 4-byte-checks,
 * counts and keeps the longest candidate within the attempt budget and the
 * compression window - the CPU ZSTD_fast "longest in chain order" semantics.
 */
__global__
static void ZSTD_gpu_searchMatches(unsigned char const* d_window, unsigned long long const* __restrict__ d_sorted,
                                   unsigned const* __restrict__ d_slot, unsigned const* __restrict__ d_rowStart,
                                   unsigned const* __restrict__ d_rowEnd, ZSTD_GpuPosCandidate* d_cand,
                                   const unsigned hashLog, const unsigned searchLog, const unsigned minMatch,
                                   const unsigned nValid, const size_t srcSize, const unsigned maxDistance)
{
    __shared__ unsigned s_matchCount;
    if (threadIdx.x == 0) s_matchCount = 0u;
    unsigned const p = (unsigned)(blockIdx.x * (size_t)blockDim.x + threadIdx.x);

    /* Candidates live at d_cand[1..nValid]; d_cand[0] is the block's
     * match counter (written once per block, read by the host to decide
     * whether the greedy scan has anything to do). */
    unsigned bestOff = 0u;
    unsigned bestLen = 0u;
    if (p < nValid)
    {
        unsigned const h = ZSTD_gpu_hashPtr(d_window + p, hashLog);
        unsigned const rs = d_rowStart[h];
        unsigned const re = d_rowEnd[h];
        unsigned long long const maxDist = (maxDistance >= 32) ? ~0ULL : (1ULL << maxDistance);

        if (re >= rs) /* p's own row is always non-empty (it contains p) */
        {
            unsigned long long const cur = (unsigned long long) p;
            unsigned attempts = 1u << searchLog;
            unsigned j = d_slot[p];
            while (j > rs && attempts > 0)
            {
                unsigned const q = (unsigned)(d_sorted[j - 1] & ZSTD_GPU_POS_MASK);
                if (cur - (unsigned long long) q > maxDist) break; /* row ascending : the rest is further */
                if (ZSTD_gpu_read32(d_window + p) == ZSTD_gpu_read32(d_window + q))
                {
                    size_t const ml = ZSTD_gpu_count(d_window + p, d_window + q, d_window + srcSize, ZSTD_GPU_MATCH_CAP);
                    if (ml > bestLen)
                    {
                        bestLen = (unsigned) ml;
                        bestOff = p - q;
                    }
                }
                j--;
                attempts--;
            }
        }
        d_cand[p + 1].offset = bestOff;
        d_cand[p + 1].matchLength = bestLen;
        if (bestOff > 0u) atomicAdd(&s_matchCount, 1u);
    }
    __syncthreads();
    if (threadIdx.x == 0 && s_matchCount > 0u) atomicAdd(&d_cand[0].offset, s_matchCount);
}

/**
 * G6 : sequential greedy resolution, done on the host. Walks the per-position
 * candidates in order and keeps the first non-overlapping one, reproducing the
 * CPU greedy "non-overlapping, longest-in-chain-order" sequence output.
 *
 * Why host side : a device-side sequential pass is latency bound - every
 * position is a dependent global load, and a block with few or no matches
 * (random data) pays the full 128 K-position walk on a single thread (~tens
 * of ms). The candidate array is dense (8 bytes per position, at most 1 MB
 * for a full block), so a D2H copy plus a streaming host scan is much faster
 * (this is the "sequential host pass over GPU output" option of the G6 spec).
 *
 * cand[0] is the block's match counter : when it is zero (incompressible
 * block) the whole scan is skipped. A candidate whose length reached the
 * GPU cap (ZSTD_GPU_MATCH_CAP) is truncated : the full length is recovered
 * here, on the host, where the source buffer is (the CPU matchfinder counts
 * the full length, so the sequences stay length-identical to CPU greedy).
 */
static void ZSTD_gpu_resolveGreedy_host(ZSTD_GpuMatchResult* const results,
                                        ZSTD_GpuPosCandidate const* const cand, void const* src,
                                        const size_t srcSize, const unsigned minMatch)
{
    if (cand[0].offset == 0u)
    {
        results->nbSeq = 0;
        return;
    }
    BYTE const* const srcB = (BYTE const*) src;
    ZSTD_GpuPosCandidate const* const pos_cand = cand + 1; /* skip the counter slot */
    size_t pos = 0;
    size_t nbSeq = 0;

    while (pos + minMatch <= srcSize && nbSeq < ZSTD_GPU_MAX_SEQUENCES)
    {
        ZSTD_GpuPosCandidate const c = pos_cand[pos];
        if (c.offset > 0 && (size_t) c.matchLength >= minMatch)
        {
            size_t const remaining = srcSize - pos;
            size_t ml = (size_t) c.matchLength;
            if (ml >= ZSTD_GPU_MATCH_CAP) /* truncated by the GPU : extend it */
            {
                while (ml < remaining && srcB[pos + ml] == srcB[pos + ml - c.offset])
                    ml++;
            }
            if (ml > remaining) ml = remaining;
            if (ml >= minMatch)
            {
                results->sequences[nbSeq].position = (unsigned) pos;
                results->sequences[nbSeq].offset = c.offset;
                results->sequences[nbSeq].matchLength = (unsigned) ml;
                nbSeq++;
                pos += ml;
                continue;
            }
        }
        pos++;
    }
    results->nbSeq = (unsigned) nbSeq;
}

/* ======================================================================== */
/* Host-Side Implementation                                                   */
/* ======================================================================== */
/* Note: ZSTD_GpuMatchFinder_s is fully defined in zstd_cuda_matchfinder.h.
 * The host API below is plain C (called from the C wrapper and from
 * zstd_compress.c / programs/fileio.c) : it needs C linkage, otherwise the
 * C++ compiler mangles the names and the link fails. */

extern "C" {

/**
 * Check if CUDA is available.
 * @return 1 if CUDA is available, 0 otherwise.
 */
int ZSTD_cudaAvailable(void)
{
    int deviceCount = 0;
    return cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0 ? 1 : 0;
}

/**
 * Run the GPU match finding pipeline for one block.
 * On return, gpuMF->h_results holds the greedy non-overlapping sequences.
 */
static size_t ZSTD_runGpuMatchFinding(
    ZSTD_GpuMatchFinder* gpuMF,
    void const* src,
    const size_t srcSize)
{
    if (!gpuMF || !src) return ERROR(GPU_invalidParameter);

    unsigned const hashLog = gpuMF->params.hashLog;
    unsigned const searchLog = gpuMF->params.searchLog;
    unsigned const minMatch = gpuMF->params.minMatch;

    /* Enforce the CCtx compression window : the maximum match distance is
     * windowSize (set by the wrapper from zc->appliedParams.cParams.windowSize),
     * not a value derived from srcSize. Matches beyond the window would make
     * the frame undecodable. Falls back to the srcSize-derived value if the
     * window was not set. */
    unsigned windowLog = 0;
    {
        size_t const w = (gpuMF->windowSize > 0) ? gpuMF->windowSize : srcSize;
        size_t s = w;
        while (s > 1)
        {
            s >>= 1;
            windowLog++;
        }
        if (windowLog > 27) windowLog = 27; /* cap at zstd max */
    }
    unsigned const maxDistance = windowLog;

    /* Number of positions with a 4-byte fingerprint (rows of the table) */
    unsigned const nValid = (srcSize >= 4) ? (unsigned)(srcSize - 3) : 0;
    /* Padded array size : a multiple of the radix block size so the grid is
     * uniform. The padding keys (~0ULL) sort after every real key. */
    unsigned const n2 = (nValid + ZSTD_GPU_BLOCK_THREADS - 1) & ~(ZSTD_GPU_BLOCK_THREADS - 1);

    if (nValid == 0)
    {
        gpuMF->h_results->nbSeq = 0;
        return 0;
    }

    /* 1) fill composite keys. Reads the device window (the H2D copy made by
     * the caller), not the host src. */
    {
        unsigned const blocks = n2 / ZSTD_GPU_BLOCK_THREADS;
        ZSTD_gpu_fillKeys<<<blocks, ZSTD_GPU_BLOCK_THREADS, 0, gpuMF->stream>>>(gpuMF->d_window,
            (unsigned long long*) gpuMF->d_keyA, hashLog, nValid, n2);
    }

    /* 2) stable LSD radix sort over the digits that are not already ordered
     * by the fill (bits 16-47, 4 passes of 8 bits), ping-ponging A/B.
     * After an even number of passes the sorted array is back in d_keyA. */
    {
        unsigned long long* a = (unsigned long long*) gpuMF->d_keyA;
        unsigned long long* b = (unsigned long long*) gpuMF->d_keyB;
        unsigned const blocks = n2 / ZSTD_GPU_BLOCK_THREADS;
        for (unsigned pass = 0; pass < ZSTD_GPU_RADIX_PASSES; pass++)
        {
            unsigned const shift = 16 + pass * ZSTD_GPU_RADIX_BITS;
            cudaMemsetAsync(gpuMF->d_blockHist, 0, (size_t) blocks * ZSTD_GPU_RADIX_BINS * sizeof(unsigned), gpuMF->stream);
            ZSTD_gpu_radixHist<<<blocks, ZSTD_GPU_BLOCK_THREADS, 0, gpuMF->stream>>>(a, (unsigned*) gpuMF->d_blockHist, shift, n2);
            ZSTD_gpu_radixScan<<<1, ZSTD_GPU_RADIX_BINS, 0, gpuMF->stream>>>((unsigned*) gpuMF->d_blockHist, (unsigned*) gpuMF->d_digitStart, blocks);
            ZSTD_gpu_radixScatter<<<blocks, ZSTD_GPU_BLOCK_THREADS, 0, gpuMF->stream>>>(a, b, (unsigned*) gpuMF->d_blockHist, (unsigned*) gpuMF->d_digitStart, shift, n2);
            unsigned long long* const tmp = a; a = b; b = tmp;
        }
        (void) b;
    }

    {
        unsigned long long* const d_sorted = (unsigned long long*) gpuMF->d_keyA;
        /* 3) slot[position] = index in the sorted array */
        {
            unsigned const blocks = n2 / ZSTD_GPU_BLOCK_THREADS;
            ZSTD_gpu_recordSlot<<<blocks, ZSTD_GPU_BLOCK_THREADS, 0, gpuMF->stream>>>(d_sorted, (unsigned*) gpuMF->d_slot, n2);
        }
        /* 4) CSR row bounds */
        {
            unsigned const blocks = n2 / ZSTD_GPU_BLOCK_THREADS;
            ZSTD_gpu_rowBounds<<<blocks, ZSTD_GPU_BLOCK_THREADS, 0, gpuMF->stream>>>(d_sorted, (unsigned*) gpuMF->d_rowStart,
                (unsigned*) gpuMF->d_rowEnd, hashLog, n2);
        }
        /* 5) parallel search */
        {
            unsigned const blocks = (nValid + ZSTD_GPU_BLOCK_THREADS - 1) / ZSTD_GPU_BLOCK_THREADS;
            /* zero the match-counter slot (d_cand[0]) */
            cudaMemsetAsync(gpuMF->d_cand, 0, sizeof(ZSTD_GpuPosCandidate), gpuMF->stream);
            ZSTD_gpu_searchMatches<<<blocks, ZSTD_GPU_BLOCK_THREADS, 0, gpuMF->stream>>>(gpuMF->d_window, d_sorted,
                (unsigned*) gpuMF->d_slot, (unsigned*) gpuMF->d_rowStart, (unsigned*) gpuMF->d_rowEnd,
                (ZSTD_GpuPosCandidate*) gpuMF->d_cand, hashLog, searchLog, minMatch, nValid, srcSize, maxDistance);
        }
    }

    /* 6) bring the candidates back (the counter rides in d_cand[0]) and
     * resolve greedily on the host */
    if (const cudaError_t syncErr = cudaStreamSynchronize(gpuMF->stream); syncErr != cudaSuccess)
    {
        fprintf(stderr, "GPU match finding: cudaStreamSynchronize failed: %s (code=%d)\n", cudaGetErrorString(syncErr),
                (int) syncErr);
        return ERROR(GPU_cudaMemcpyFailed);
    }
    {
        size_t const candBytes = ((size_t) nValid + 1) * sizeof(ZSTD_GpuPosCandidate);
        if (const cudaError_t copyErr = cudaMemcpy(gpuMF->cand_host, gpuMF->d_cand, candBytes, cudaMemcpyDeviceToHost);
            copyErr != cudaSuccess)
        {
            fprintf(stderr, "GPU match finding: candidates D2H failed: %s (code=%d)\n", cudaGetErrorString(copyErr), (int) copyErr);
            return ERROR(GPU_cudaMemcpyFailed);
        }
    }
    ZSTD_gpu_resolveGreedy_host(gpuMF->h_results, (ZSTD_GpuPosCandidate const*) gpuMF->cand_host, src, srcSize, minMatch);

    return 0;
}

/* ======================================================================== */
/* API : create / free / estimate                                            */
/* ======================================================================== */

size_t ZSTD_createGpuMatchFinder(ZSTD_GpuMatchFinder** gpuMF, const ZSTD_GpuCompressionParameters* cParams,
                                 const size_t srcSize)
{
    if (!gpuMF || !cParams || !srcSize) return ERROR(GPU_invalidParameter);

    ZSTD_GpuMatchFinder* ctx = (ZSTD_GpuMatchFinder*) malloc(sizeof(ZSTD_GpuMatchFinder));
    if (!ctx) return ERROR(GPU_allocationFailed);

    memset(ctx, 0, sizeof(ZSTD_GpuMatchFinder));
    ctx->params = *cParams;
    ctx->windowSize = srcSize + ZSTD_BLOCKSIZE_MAX;
    ctx->initialized = 1;

    /* Validation : reject parameters that would require unrealistic memory */
    if (cParams->hashLog < 6 || cParams->hashLog > ZSTD_GPU_HASHLOG_MAX)
    {
        free(ctx);
        return ERROR(GPU_invalidParameter);
    }
    if (cParams->minMatch < 3 || cParams->minMatch > 7)
    {
        free(ctx);
        return ERROR(GPU_invalidParameter);
    }

    /* Create CUDA stream */
    cudaError_t err = cudaStreamCreate(&ctx->stream);
    if (err != cudaSuccess)
    {
        free(ctx);
        return ERROR(GPU_cudaStreamCreationFailed);
    }

    /* Allocate device memory :
     *   - sliding window (source data)
     *   - 2 sort-key ping-pong buffers (128 KB x 8 B each)
     *   - per-position slot indices (128 KB x 4 B)
     *   - CSR row start/end (2^hashLog x 4 B each)
     *   - per-position candidates (8 B each + 1 counter slot)
     *   - radix per-block histogram (512 blocks x 256 bins x 4 B = 512 KB)
     */
    size_t const hashTableSize = static_cast<size_t>(1) << cParams->hashLog;

    if (cudaMalloc(&ctx->d_window, srcSize + ZSTD_BLOCKSIZE_MAX) != cudaSuccess
        || cudaMalloc(&ctx->d_keyA, ZSTD_GPU_BLOCKSIZE_MAX * sizeof(unsigned long long)) != cudaSuccess
        || cudaMalloc(&ctx->d_keyB, ZSTD_GPU_BLOCKSIZE_MAX * sizeof(unsigned long long)) != cudaSuccess
        || cudaMalloc(&ctx->d_slot, ZSTD_GPU_BLOCKSIZE_MAX * sizeof(unsigned)) != cudaSuccess
        || cudaMalloc(&ctx->d_rowStart, hashTableSize * sizeof(unsigned)) != cudaSuccess
        || cudaMalloc(&ctx->d_rowEnd, hashTableSize * sizeof(unsigned)) != cudaSuccess
        || cudaMalloc(&ctx->d_cand, (ZSTD_GPU_BLOCKSIZE_MAX + 1) * sizeof(ZSTD_GpuPosCandidate)) != cudaSuccess
        || cudaMalloc(&ctx->d_blockHist, ZSTD_GPU_BLOCKSIZE_MAX * sizeof(unsigned)) != cudaSuccess
        || cudaMalloc(&ctx->d_digitStart, 256 * sizeof(unsigned)) != cudaSuccess)
    {
        /* release everything allocated so far (ctx is zero-initialized, so
         * cudaFree(NULL) on slots that failed is safe) */
        cudaFree(ctx->d_window);
        cudaFree(ctx->d_keyA);
        cudaFree(ctx->d_keyB);
        cudaFree(ctx->d_slot);
        cudaFree(ctx->d_rowStart);
        cudaFree(ctx->d_rowEnd);
        cudaFree(ctx->d_cand);
        cudaFree(ctx->d_blockHist);
        cudaFree(ctx->d_digitStart);
        cudaStreamDestroy(ctx->stream);
        free(ctx);
        return ERROR(GPU_cudaAllocationFailed);
    }

    /* Pinned host buffer for the candidates (D2H target of the host-side
     * greedy resolution) */
    if (cudaMallocHost(&ctx->cand_host, ZSTD_GPU_BLOCKSIZE_MAX * sizeof(ZSTD_GpuPosCandidate)) != cudaSuccess)
    {
        cudaStreamDestroy(ctx->stream);
        free(ctx);
        return ERROR(GPU_cudaAllocationFailed);
    }

    /* Pinned host result buffer */
    if (cudaMallocHost((void**) &ctx->h_results, sizeof(ZSTD_GpuMatchResult)) != cudaSuccess)
    {
        cudaStreamDestroy(ctx->stream);
        free(ctx);
        return ERROR(GPU_cudaAllocationFailed);
    }

    *gpuMF = ctx;
    return 0;
}

size_t ZSTD_freeGpuMatchFinder(ZSTD_GpuMatchFinder* gpuMF)
{
    if (!gpuMF) return 0;

    cudaStreamSynchronize(gpuMF->stream);
    cudaStreamDestroy(gpuMF->stream);

    cudaFree(gpuMF->d_window);
    cudaFree(gpuMF->d_keyA);
    cudaFree(gpuMF->d_keyB);
    cudaFree(gpuMF->d_slot);
    cudaFree(gpuMF->d_rowStart);
    cudaFree(gpuMF->d_rowEnd);
    cudaFree(gpuMF->d_cand);
    cudaFree(gpuMF->d_blockHist);
    cudaFree(gpuMF->d_digitStart);
    cudaFreeHost(gpuMF->cand_host);
    if (gpuMF->h_results) cudaFreeHost(gpuMF->h_results);

    free(gpuMF);
    return 0;
}

size_t ZSTD_estimateGpuMatchFinderMemory(const ZSTD_GpuCompressionParameters* params, const size_t srcSize)
{
    if (!params || !srcSize) return 0;

    size_t total = 0;
    total += srcSize + ZSTD_BLOCKSIZE_MAX;  /* Sliding window */
    total += ZSTD_GPU_BLOCKSIZE_MAX * sizeof(unsigned long long) * 2; /* Sort key ping-pong buffers */
    total += ZSTD_GPU_BLOCKSIZE_MAX * sizeof(unsigned) * 2;           /* Slot indices + radix block histogram */
    total += 256 * sizeof(unsigned);                                  /* Radix digit starts */
    total += (static_cast<size_t>(1) << params->hashLog) * sizeof(unsigned) * 2; /* CSR row start/end */
    total += (ZSTD_GPU_BLOCKSIZE_MAX + 1) * sizeof(ZSTD_GpuPosCandidate) * 2; /* Candidates device + pinned host */
    total += sizeof(ZSTD_GpuMatchResult); /* Pinned host results */

    return total;
}

/* ======================================================================== */
/* Compress Block GPU Entry Point                                            */
/* ======================================================================== */

/**
 * Convert GPU match results to ZSTD_Sequence format (defined below).
 */
static size_t ZSTD_convertResultsToSequences(ZSTD_GpuMatchFinder* gpuMF, void const* src, size_t srcSize,
                                             ZSTD_Sequence* dst, size_t dstCapacity);

/**
 * Compress one block with GPU match finding (entry point for the wrapper).
 *
 * @param gpuMF GPU match finder context
 * @param dst[out] Output buffer for ZSTD_Sequence results
 * @param dstCapacity Capacity of dst, in sequences
 * @param src Input block
 * @param srcSize Size of src, in bytes (<= ZSTD_BLOCKSIZE_MAX)
 * @param rep In/out repcode array (unused by the GPU : always 0)
 * @param repOut In/out repcode array (unused by the GPU : always 0)
 * @return Number of sequences written, or an error code
 */
size_t ZSTD_compressBlock_gpu(
    ZSTD_GpuMatchFinder* gpuMF,
    ZSTD_Sequence* dst,
    const size_t dstCapacity,
    void const* src,
    const size_t srcSize,
    U32* rep,
    U32* repOut)
{
    (void) rep;
    (void) repOut;

    if (!gpuMF || !dst || !src) return ERROR(GPU_invalidParameter);

    /* Copy source to GPU window */
    cudaError_t err = cudaMemcpyAsync(gpuMF->d_window, src, srcSize, cudaMemcpyHostToDevice, gpuMF->stream);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "GPU match finding: H2D failed: %s (code=%d)\n", cudaGetErrorString(err), (int) err);
        return ERROR(GPU_cudaMemcpyFailed);
    }

    /* Run GPU match finding */
    {
        size_t const runErr = ZSTD_runGpuMatchFinding(gpuMF, src, srcSize);
        if (ZSTD_isError(runErr)) return runErr;
    }

    /* Convert results to sequences */
    size_t const nbSeq = ZSTD_convertResultsToSequences(gpuMF, src, srcSize, dst, dstCapacity);
    if (ZSTD_isError(nbSeq)) return nbSeq;

    return nbSeq;
}

/**
 * Convert GPU match results to ZSTD_Sequence format.
 */
static size_t ZSTD_convertResultsToSequences(ZSTD_GpuMatchFinder* gpuMF, void const* src, size_t srcSize,
                                             ZSTD_Sequence* dst, size_t dstCapacity)
{
    (void) src;
    if (!gpuMF || !dst) return ERROR(GPU_invalidParameter);

    ZSTD_GpuMatchResult* results = gpuMF->h_results;
    if (!results) return ERROR(GPU_allocationFailed);

    size_t seqIdx = 0;
    size_t pos = 0;

    for (unsigned i = 0; i < results->nbSeq && seqIdx < dstCapacity; i++)
    {
        ZSTD_GpuSequence const* seq = &results->sequences[i];
        size_t const matchPos = seq->position;
        size_t const matchLen = seq->matchLength;
        unsigned const offset = seq->offset;

        if (matchPos < pos) continue; /* Overlap guard */

        /* Literals length from current position to match start */
        size_t const litLen = matchPos - pos;

        /* Clamp match length to block end */
        size_t const remaining = (srcSize > matchPos) ? srcSize - matchPos : 0;
        size_t const mLen = (matchLen > remaining) ? remaining : matchLen;
        if (mLen < MINMATCH) continue;

        if (seqIdx < dstCapacity)
        {
            dst[seqIdx].litLength = (unsigned) litLen;
            dst[seqIdx].matchLength = (unsigned) mLen;
            dst[seqIdx].offset = offset;
            dst[seqIdx].rep = 0;
        }
        seqIdx++;
        pos = matchPos + mLen;
    }

    return seqIdx;
}

} /* extern "C" */
