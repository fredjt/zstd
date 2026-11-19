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
 * @brief GPU-accelerated match finding for zstd (POC).
 *
 * Architecture (batched + windowed + pipelined):
 *
 *  - The device holds a sliding window (up to 1 MB) of input bytes and a
 *    persistent hash table (fingerprint -> latest position, stored as
 *    position+1 with 0 meaning empty). Table state carries across 128 KB
 *    frame blocks, mirroring the CPU match state's window semantics, so
 *    matches may cross block boundaries (G14).
 *
 *  - Blocks are processed in batches of B. Per batch:
 *      copy stream : [window memmove] + H2D append of the batch bytes
 *      compute     : wait -> fillKeys -> nPasses x radix -> recordSlot ->
 *                    rowBounds -> search (candidate = max(in-batch row
 *                    predecessor, pre-batch table entry)) -> insert
 *                    (atomicMax table update for the next batch)
 *      copy stream : wait -> D2H of the per-position candidates
 *    Pinned staging is double-buffered so the H2D of batch k+1 overlaps the
 *    compute of batch k (G10). One sync per batch, not per block (G11).
 *
 *  - The search uses the CPU's exact fingerprint hash (minMatch-dependent),
 *    an 8-byte prefilter and a window-age check before the per-position count
 *    (G12). Greedy sequence resolution (non-overlapping, in position order)
 *    currently runs on the host over the D2H'd candidates (G13 moves it to
 *    the device).
 *
 * See doc/gpu_matchfinding_design.md for the full design rationale and the
 * tradeoffs vs. the CPU match finder.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "zstd_cuda_matchfinder.h"

/* ======================================================================== */
/* Constants                                                                 */
/* ======================================================================== */

#define ZSTD_GPU_BLOCKSIZE      131072u                  /* 128 KB frame block */
#define ZSTD_GPU_MAX_SEQS       (ZSTD_GPU_BLOCKSIZE / 4) /* 32768 seqs/block max */
#define ZSTD_GPU_MATCH_CAP      8u                       /* search count cap (bytes) */
#define ZSTD_GPU_THREADS        256u                     /* threads per CUDA block */
#define ZSTD_GPU_MAX_POSBITS    23u                      /* bits for B*BLOCKSIZE (B=64) */

/* ======================================================================== */
/* Device helpers                                                            */
/* ======================================================================== */

__device__ __forceinline__ unsigned long long gpuRead64(const unsigned char* p)
{
    unsigned long long v;
    memcpy(&v, p, 8);
    return v;
}

__device__ __forceinline__ unsigned gpuRead32(const unsigned char* p)
{
    unsigned v;
    memcpy(&v, p, 4);
    return v;
}

/* The CPU's exact fingerprint hash (zstd_compress_internal.h ZSTD_hashPtr),
 * minMatch-dependent. Reproduced bit-for-bit so the GPU candidate set matches
 * the CPU fast match finder's. */
__device__ __forceinline__ unsigned gpuHash(const unsigned char* p, unsigned hBits, unsigned mls)
{
    if (mls == 4) {
        return (unsigned)(((unsigned long long)gpuRead32(p) * 2654435761ULL) >> (32 - hBits));
    }
    {
        unsigned long long const u = gpuRead64(p);
        switch (mls) {
            case 5:  return (unsigned)((((u << (64-40)) * 889523592379ULL)      >> (64 - hBits)));
            case 6:  return (unsigned)((((u << (64-48)) * 227718039650203ULL)   >> (64 - hBits)));
            case 7:  return (unsigned)((((u << (64-56)) * 58295818150454627ULL) >> (64 - hBits)));
            default: return (unsigned)((((u << (64-40)) * 889523592379ULL)      >> (64 - hBits)));
        }
    }
}

/* Map a batch position t (0..totalPos-1) to (block i, absolute position a,
 * position-within-block pib). A block of `size` bytes contributes (size-7)
 * valid positions (the last 7 cannot start an 8-byte fingerprint). */
__device__ void gpuBatchLocate(const unsigned* d_blockMeta, unsigned B, unsigned t,
                               unsigned& i, unsigned& a, unsigned& pib)
{
    for (i = 0; ; i++) {
        unsigned const size = d_blockMeta[i*2 + 1];
        unsigned const n = (size >= 8) ? (size - 7) : 0;
        if (t < n) {
            pib = t;
            a = d_blockMeta[i*2] + t;
            return;
        }
        t -= n;
    }
}

/* ======================================================================== */
/* Kernels                                                                   */
/* ======================================================================== */

/* One thread per valid position: write the composite sort key
 * key = (hash << posBits) | t.  The low posBits bits are the in-batch
 * position (already in order), the high bits are the fingerprint. */
__global__ void gpuFillKeys(const unsigned char* __restrict__ d_window, unsigned winBase,
                            const unsigned* __restrict__ d_blockMeta, unsigned B,
                            unsigned hashLog, unsigned mls, unsigned posBits,
                            unsigned long long* __restrict__ d_keys, unsigned totalPos)
{
    unsigned const t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= totalPos) return;
    unsigned i, a, pib;
    gpuBatchLocate(d_blockMeta, B, t, i, a, pib);
    unsigned char const* const p = d_window + (a - winBase);
    unsigned const h = gpuHash(p, hashLog, mls);
    d_keys[t] = ((unsigned long long)h << posBits) | (unsigned long long)t;
}

/* Per-CUDA-block histogram of the current digit (8 bits). */
__global__ void gpuRadixHist(const unsigned long long* __restrict__ d_keys,
                             unsigned bitOffset, unsigned totalPos,
                             unsigned* __restrict__ d_blockHist)
{
    __shared__ unsigned s_hist[256];
    unsigned const t = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < 256) s_hist[threadIdx.x] = 0;
    __syncthreads();
    if (t < totalPos) {
        unsigned const d = (unsigned)((d_keys[t] >> bitOffset) & 255);
        atomicAdd(&s_hist[d], 1);
    }
    __syncthreads();
    if (threadIdx.x < 256) d_blockHist[blockIdx.x * 256 + threadIdx.x] = s_hist[threadIdx.x];
}

/* Level-1 scan: per group of 256 CUDA blocks, per digit, the exclusive
 * within-group prefix (d_blockStartLocal) and the group total (d_groupTotal). */
__global__ void gpuRadixScanL1(const unsigned* __restrict__ d_blockHist,
                               unsigned nBlocks,
                               unsigned* __restrict__ d_blockStartLocal,
                               unsigned* __restrict__ d_groupTotal)
{
    unsigned const d = threadIdx.x;          /* digit */
    unsigned const g = blockIdx.x;           /* group */
    unsigned const cb0 = g * 256;
    unsigned const cb1 = min(cb0 + 256, nBlocks);
    unsigned running = 0;
    for (unsigned cb = cb0; cb < cb1; cb++) {
        d_blockStartLocal[cb * 256 + d] = running;
        running += d_blockHist[cb * 256 + d];
    }
    d_groupTotal[g * 256 + d] = running;
}

/* Level-2 scan: digit totals, the digit prefix (d_digitStart), and the group
 * prefix (d_groupPrefix). */
__global__ void gpuRadixScanL2(const unsigned* __restrict__ d_groupTotal,
                               unsigned nGroups,
                               unsigned* __restrict__ d_digitTotal,
                               unsigned* __restrict__ d_digitStart,
                               unsigned* __restrict__ d_groupPrefix)
{
    unsigned const d = threadIdx.x;
    unsigned total = 0;
    for (unsigned g = 0; g < nGroups; g++) total += d_groupTotal[g * 256 + d];
    d_digitTotal[d] = total;
    __syncthreads();
    if (threadIdx.x == 0) {
        d_digitStart[0] = 0;
        for (unsigned i = 1; i < 256; i++) d_digitStart[i] = d_digitStart[i-1] + d_digitTotal[i-1];
    }
    __syncthreads();
    {
        unsigned running = 0;
        for (unsigned g = 0; g < nGroups; g++) {
            d_groupPrefix[g * 256 + d] = running;
            running += d_digitTotal[d];
        }
    }
}

/* Scatter: place each key into d_keysB at its sorted position.
 * dst = digitStart[d] + groupPrefix[g][d] + blockStartLocal[cb][d] + localRank.
 * The local rank within (cb, d) is an atomic counter (zeroed per pass). */
__global__ void gpuRadixScatter(const unsigned long long* __restrict__ d_keys,
                                unsigned bitOffset, unsigned totalPos,
                                const unsigned* __restrict__ d_digitStart,
                                const unsigned* __restrict__ d_groupPrefix,
                                const unsigned* __restrict__ d_blockStartLocal,
                                unsigned* __restrict__ d_blockCounter,
                                unsigned long long* __restrict__ d_keysB)
{
    unsigned const t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= totalPos) return;
    unsigned long long const key = d_keys[t];
    unsigned const d = (unsigned)((key >> bitOffset) & 255);
    {
        unsigned const g = blockIdx.x / 256;
        unsigned const localRank = atomicAdd(&d_blockCounter[blockIdx.x * 256 + d], 1);
        unsigned const dst = d_digitStart[d] + d_groupPrefix[g * 256 + d]
                           + d_blockStartLocal[blockIdx.x * 256 + d] + localRank;
        d_keysB[dst] = key;
    }
}

/* slot[t] = index of key t in the sorted array. */
__global__ void gpuRecordSlot(const unsigned long long* __restrict__ d_keysSorted,
                              unsigned posBits, unsigned totalPos,
                              unsigned* __restrict__ d_slot)
{
    unsigned const j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= totalPos) return;
    unsigned const t = (unsigned)(d_keysSorted[j] & ((1ull << posBits) - 1));
    d_slot[t] = j;
}

/* Row bounds per hash over the whole batch: the sorted array is ordered by
 * (hash, position), so each hash's row is contiguous. Each row's start/end is
 * written by exactly one thread (the boundary element). Empty rows are
 * pre-initialized to [totalPos, totalPos). */
__global__ void gpuRowBoundsInit(unsigned* __restrict__ d_rowStart,
                                 unsigned* __restrict__ d_rowEnd,
                                 unsigned hashLog, unsigned totalPos)
{
    unsigned const h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h < (1u << hashLog)) { d_rowStart[h] = totalPos; d_rowEnd[h] = totalPos; }
}

__global__ void gpuRowBounds(const unsigned long long* __restrict__ d_keysSorted,
                             unsigned posBits, unsigned totalPos,
                             unsigned* __restrict__ d_rowStart,
                             unsigned* __restrict__ d_rowEnd)
{
    unsigned const j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= totalPos) return;
    unsigned const h = (unsigned)(d_keysSorted[j] >> posBits);
    if (j == 0) {
        d_rowStart[h] = j;
    } else {
        unsigned const hPrev = (unsigned)(d_keysSorted[j-1] >> posBits);
        if (hPrev != h) {
            d_rowEnd[hPrev] = j;
            d_rowStart[h] = j;
        }
    }
    if (j == totalPos - 1) d_rowEnd[h] = j + 1;
}

/* The search: for each valid position, the candidate is the max of the
 * in-batch row predecessor (the closest previous same-hash position in the
 * batch) and the pre-batch table entry (the latest same-hash position before
 * the batch). A candidate must be within the window and pass an 8-byte
 * prefilter before the (capped) count. The result is written directly to
 * d_cand[block][posInBlock] = (offset, len); offset == 0 means no candidate. */
__global__ void gpuSearch(const unsigned char* __restrict__ d_window, unsigned winBase,
                          const unsigned* __restrict__ d_blockMeta, unsigned B,
                          const unsigned long long* __restrict__ d_keysSorted,
                          const unsigned* __restrict__ d_slot,
                          const unsigned* __restrict__ d_table,
                          unsigned hashLog, unsigned mls, unsigned posBits,
                          unsigned windowSize, unsigned minMatch,
                          unsigned totalPos,
                          unsigned* __restrict__ d_cand)
{
    unsigned const t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= totalPos) return;
    unsigned i, a, pib;
    gpuBatchLocate(d_blockMeta, B, t, i, a, pib);
    unsigned char const* const p = d_window + (a - winBase);
    unsigned const h = gpuHash(p, hashLog, mls);

    /* In-batch predecessor: the element just before t in the sorted array, if
     * it has the same hash. Position 0 is a valid absolute position, so track
     * validity separately. */
    unsigned q1 = 0; int hasQ1 = 0;
    {
        unsigned const j = d_slot[t];
        if (j > 0) {
            unsigned long long const prev = d_keysSorted[j-1];
            if ((unsigned)(prev >> posBits) == h) {
                unsigned const tPrev = (unsigned)(prev & ((1ull << posBits) - 1));
                unsigned iPrev, aPrev, pibPrev;
                gpuBatchLocate(d_blockMeta, B, tPrev, iPrev, aPrev, pibPrev);
                q1 = aPrev; hasQ1 = 1;
            }
        }
    }
    /* Pre-batch table entry (latest same-hash position before the batch),
     * stored as pos+1 with 0 meaning empty. */
    unsigned q2 = 0; int hasQ2 = 0;
    if (d_table[h] != 0) { q2 = d_table[h] - 1; hasQ2 = 1; }

    /* Candidate = the max of the valid entries (both are < a). */
    unsigned qc = 0; int hasQ = 0;
    if (hasQ1) { qc = q1; hasQ = 1; }
    if (hasQ2 && (!hasQ1 || q2 > qc)) { qc = q2; hasQ = 1; }
    if (!hasQ) return;

    /* Window-age check (G12.2): skip candidates older than the window before
     * touching the data. */
    if (a - qc > windowSize) return;

    unsigned char const* const qp = d_window + (qc - winBase);
    /* 8-byte prefilter (G12.1): two 64-bit loads at the candidate head. Valid
     * positions always have 8 bytes left in the block (nValid = size-7), so the
     * read stays inside the block. For the GPU levels minMatch <= 8, so a
     * passing prefilter means the candidate is at least minMatch long; the
     * capped length is ZSTD_GPU_MATCH_CAP and the greedy extends it. */
    (void)minMatch;
    if (gpuRead64(p) != gpuRead64(qp)) return;

    unsigned* const c = d_cand + (i * ZSTD_GPU_BLOCKSIZE + pib) * 2;
    c[0] = a - qc;                    /* offset */
    c[1] = ZSTD_GPU_MATCH_CAP;        /* capped length (the greedy extends it) */
}

/* Insert: update the persistent table with the batch's positions (the latest
 * position per fingerprint). Runs after the search so the search saw the
 * pre-batch table. */
__global__ void gpuInsert(const unsigned char* __restrict__ d_window, unsigned winBase,
                          const unsigned* __restrict__ d_blockMeta, unsigned B,
                          unsigned hashLog, unsigned mls,
                          unsigned totalPos,
                          unsigned* __restrict__ d_table)
{
    unsigned const t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= totalPos) return;
    unsigned i, a, pib;
    gpuBatchLocate(d_blockMeta, B, t, i, a, pib);
    unsigned char const* const p = d_window + (a - winBase);
    unsigned const h = gpuHash(p, hashLog, mls);
    atomicMax(d_table + h, a + 1);   /* store pos+1; a+1 is the latest so far */
}

/* ======================================================================== */
/* Device greedy : one thread per block scans its candidates and extends the
 * matches using the window (so cross-block matches are extended correctly).
 * ======================================================================== */
__global__ void gpuGreedy(const unsigned char* __restrict__ d_window, unsigned winBase,
                          const unsigned* __restrict__ d_blockMeta, unsigned B,
                          const unsigned* __restrict__ d_cand,
                          ZSTD_GpuSequence* __restrict__ d_seq,
                          unsigned* __restrict__ d_nbSeq)
{
    unsigned const i = blockIdx.x;
    if (i >= B) return;
    if (threadIdx.x != 0) return;   /* one thread per block */
    unsigned const size = d_blockMeta[i*2+1];
    unsigned const absStart = d_blockMeta[i*2];
    unsigned currentEnd = 0;
    unsigned nbSeq = 0;
    for (unsigned pib = 0; pib + 1 < size; pib++) {
        unsigned const off = d_cand[(i * ZSTD_GPU_BLOCKSIZE + pib) * 2];
        if (off == 0) continue;
        if (pib < currentEnd) continue;
        unsigned const a = absStart + pib;
        unsigned const qc = a - off;   /* match source (within the window) */
        unsigned len = ZSTD_GPU_MATCH_CAP;
        unsigned char* const base = (unsigned char*)(d_window + (a - winBase));
        unsigned char* cp = base + len;
        unsigned char* cq = (unsigned char*)(d_window + (qc - winBase)) + len;
        unsigned char* const cend = base + (size - pib);
        while (cp < cend && *cp == *cq) { cp++; cq++; len++; }
        ZSTD_GpuSequence* const s = d_seq + (i * ZSTD_GPU_MAX_SEQS + nbSeq);
        s->position = pib;
        s->offset = off;
        s->matchLength = len;
        nbSeq++;
        currentEnd = pib + len;
    }
    d_nbSeq[i] = nbSeq;
}


/* ======================================================================== */
/* Host: context management                                                  */
/* ======================================================================== */

static int g_gpuRefCount = 0;   /* POC: at most one GPU match finder per process */

int ZSTD_cudaAvailable(void)
{
    static int avail = -1;
    if (avail < 0) {
        int n = 0;
        avail = (cudaGetDeviceCount(&n) == cudaSuccess && n > 0) ? 1 : 0;
    }
    return avail;
}

size_t ZSTD_estimateGpuMatchFinderMemory(
    ZSTD_GpuCompressionParameters const* cParams,
    size_t srcSize)
{
    (void)srcSize;
    if (!cParams) return 0;
    {
        unsigned const hashLog = cParams->hashLog;
        unsigned const B = 16;   /* default batch size (informational) */
        size_t bytes = 0;
        bytes += (size_t)ZSTD_GPU_WINDOW_MAX + ZSTD_GPU_BLOCKSIZE;          /* window */
        bytes += (size_t)(1u << hashLog) * 4;                              /* table */
        bytes += (size_t)B * ZSTD_GPU_BLOCKSIZE * 8 * 2;                   /* keyA/keyB */
        bytes += (size_t)B * ZSTD_GPU_BLOCKSIZE * 4;                       /* slot */
        bytes += (size_t)(1u << hashLog) * 4 * 2;                          /* rowStart/End */
        bytes += (size_t)B * 512 * 256 * 4 * 3;                            /* hist/startLocal/counter */
        bytes += (size_t)B * ZSTD_GPU_BLOCKSIZE * 8;                       /* cand */
        bytes += (size_t)B * ZSTD_GPU_MAX_SEQS * 12;                       /* seq */
        bytes += (size_t)B * ZSTD_GPU_BLOCKSIZE * 8 * 2;                   /* pinned cand (x2) */
        bytes += (size_t)B * ZSTD_GPU_BLOCKSIZE * 2;                       /* pinned staging (x2) */
        return bytes;
    }
}

size_t ZSTD_createGpuMatchFinder(
    ZSTD_GpuMatchFinder** gpuMF,
    ZSTD_GpuCompressionParameters const* cParams,
    size_t srcSize)
{
    if (!gpuMF || !cParams) return ERROR(GPU_invalidParameter);
    *gpuMF = NULL;
    if (!ZSTD_cudaAvailable()) return ERROR(GPU_cudaNotAvailable);
    if (g_gpuRefCount >= 1) return ERROR(GPU_allocationFailed);  /* POC: one per process */

    ZSTD_GpuMatchFinder* mf = NULL;
    {
        mf = (ZSTD_GpuMatchFinder*)calloc(1, sizeof(*mf));
        if (!mf) return ERROR(GPU_allocationFailed);

        mf->params = *cParams;
        mf->hashLog = cParams->hashLog;
        mf->minMatch = cParams->minMatch ? cParams->minMatch : 5;
        mf->mls = mf->minMatch;
        if (mf->mls < 4) mf->mls = 4;
        if (mf->mls > 8) mf->mls = 8;
        /* Window: the CCtx compression window, capped at 1 MB (G14). */
        {
            size_t w = (cParams->windowLog > 0) ? ((size_t)1 << cParams->windowLog) : ZSTD_GPU_WINDOW_MAX;
            if (w > ZSTD_GPU_WINDOW_MAX) w = ZSTD_GPU_WINDOW_MAX;
            mf->windowSize = (unsigned)w;
        }
        mf->B = 16;   /* default batch size (G11.2 makes this adaptive) */
        if (mf->B > ZSTD_GPU_MAX_BATCH) mf->B = ZSTD_GPU_MAX_BATCH;

        /* posBits: bits needed for B*BLOCKSIZE positions. */
        {
            unsigned long long const maxPos = (unsigned long long)mf->B * ZSTD_GPU_BLOCKSIZE;
            unsigned pb = 0; while ((1ull << pb) < maxPos) pb++;
            mf->posBits = pb;
        }
        mf->nPasses = (mf->hashLog + 7) / 8;
        mf->nBlocksMax = mf->B * (ZSTD_GPU_BLOCKSIZE / ZSTD_GPU_THREADS);

        /* ---- Device buffers ---- */
        /* The window buffer must hold the window (windowSize) plus a full batch
         * (B blocks) so the H2D append never overflows. */
        {
            size_t const wbuf = (size_t)mf->windowSize + (size_t)mf->B * ZSTD_GPU_BLOCKSIZE;
            if (cudaMalloc(&mf->d_window, wbuf) != cudaSuccess) goto fail;
            if (cudaMalloc(&mf->d_windowTmp, wbuf) != cudaSuccess) goto fail;
        }
        if (cudaMalloc(&mf->d_table, (size_t)(1u << mf->hashLog) * 4) != cudaSuccess) goto fail;
        if (cudaMemset(mf->d_table, 0, (size_t)(1u << mf->hashLog) * 4) != cudaSuccess) goto fail;
        {
            size_t const kbytes = (size_t)mf->B * ZSTD_GPU_BLOCKSIZE * 8;
            if (cudaMalloc(&mf->d_keyA, kbytes) != cudaSuccess) goto fail;
            if (cudaMalloc(&mf->d_keyB, kbytes) != cudaSuccess) goto fail;
        }
        if (cudaMalloc(&mf->d_slot, (size_t)mf->B * ZSTD_GPU_BLOCKSIZE * 4) != cudaSuccess) goto fail;
        if (cudaMalloc(&mf->d_rowStart, (size_t)(1u << mf->hashLog) * 4) != cudaSuccess) goto fail;
        if (cudaMalloc(&mf->d_rowEnd, (size_t)(1u << mf->hashLog) * 4) != cudaSuccess) goto fail;
        {
            size_t const hbytes = (size_t)mf->nBlocksMax * 256 * 4;
            if (cudaMalloc(&mf->d_blockHist, hbytes) != cudaSuccess) goto fail;
            if (cudaMalloc(&mf->d_blockStartLocal, hbytes) != cudaSuccess) goto fail;
            if (cudaMalloc(&mf->d_blockCounter, hbytes) != cudaSuccess) goto fail;
            size_t const gbytes = (size_t)((mf->nBlocksMax + 255) / 256) * 256 * 4;
            if (cudaMalloc(&mf->d_groupTotal, gbytes) != cudaSuccess) goto fail;
            if (cudaMalloc(&mf->d_groupPrefix, gbytes) != cudaSuccess) goto fail;
            if (cudaMalloc(&mf->d_digitTotal, 256 * 4) != cudaSuccess) goto fail;
            if (cudaMalloc(&mf->d_digitStart, 256 * 4) != cudaSuccess) goto fail;
        }
        if (cudaMalloc(&mf->d_cand, (size_t)mf->B * ZSTD_GPU_BLOCKSIZE * 8) != cudaSuccess) goto fail;
        if (cudaMalloc(&mf->d_seq, (size_t)mf->B * ZSTD_GPU_MAX_SEQS * 12) != cudaSuccess) goto fail;
        if (cudaMalloc(&mf->d_nbSeq, (size_t)mf->B * 4) != cudaSuccess) goto fail;
        if (cudaMalloc(&mf->d_blockMeta, (size_t)mf->B * 8) != cudaSuccess) goto fail;

        /* ---- Host pinned buffers (double-buffered) ---- */
        {
            size_t const cbytes = (size_t)mf->B * ZSTD_GPU_BLOCKSIZE * 8;
            size_t const sbytes = (size_t)mf->B * ZSTD_GPU_BLOCKSIZE;
            if (cudaMallocHost(&mf->h_cand[0], cbytes) != cudaSuccess) goto fail;
            if (cudaMallocHost(&mf->h_cand[1], cbytes) != cudaSuccess) goto fail;
            if (cudaMallocHost(&mf->h_staging[0], sbytes) != cudaSuccess) goto fail;
            if (cudaMallocHost(&mf->h_staging[1], sbytes) != cudaSuccess) goto fail;
            if (cudaMallocHost(&mf->h_nbSeq[0], (size_t)mf->B * 4) != cudaSuccess) goto fail;
            if (cudaMallocHost(&mf->h_nbSeq[1], (size_t)mf->B * 4) != cudaSuccess) goto fail;
            {
                size_t const qbytes = (size_t)mf->B * ZSTD_GPU_MAX_SEQS * sizeof(ZSTD_GpuSequence);
                if (cudaMallocHost(&mf->h_seq[0], qbytes) != cudaSuccess) goto fail;
                if (cudaMallocHost(&mf->h_seq[1], qbytes) != cudaSuccess) goto fail;
            }
        }

        /* ---- Streams and events ---- */
        if (cudaStreamCreate(&mf->stream) != cudaSuccess) goto fail;
        if (cudaStreamCreate(&mf->copyStream) != cudaSuccess) goto fail;
        for (int b = 0; b < 2; b++) {
            if (cudaEventCreateWithFlags(&mf->evH2d[b], cudaEventDisableTiming) != cudaSuccess) goto fail;
            if (cudaEventCreateWithFlags(&mf->evCompute[b], cudaEventDisableTiming) != cudaSuccess) goto fail;
            if (cudaEventCreateWithFlags(&mf->evD2H[b], cudaEventDisableTiming) != cudaSuccess) goto fail;
            if (cudaEventCreateWithFlags(&mf->evNbSeq[b], cudaEventDisableTiming) != cudaSuccess) goto fail;
        }

        /* ---- Window / batch state ---- */
        mf->winBase = 0;
        mf->winUsed = 0;
        mf->framePos = 0;
        mf->batchCount = 0;
        mf->batchBuf = 0;
        mf->batchStagingOffset[0] = 0;
        mf->batchSeqsCopied[0] = mf->batchSeqsCopied[1] = 0;
        mf->batchIdx = 0;

        g_gpuRefCount = 1;
        *gpuMF = mf;
        return 0;
    }
fail:
    ZSTD_freeGpuMatchFinder(mf);
    return ERROR(GPU_allocationFailed);
}

size_t ZSTD_freeGpuMatchFinder(ZSTD_GpuMatchFinder* gpuMF)
{
    if (!gpuMF) return 0;
    if (gpuMF->d_window) cudaFree(gpuMF->d_window);
    if (gpuMF->d_windowTmp) cudaFree(gpuMF->d_windowTmp);
    if (gpuMF->d_table) cudaFree(gpuMF->d_table);
    if (gpuMF->d_keyA) cudaFree(gpuMF->d_keyA);
    if (gpuMF->d_keyB) cudaFree(gpuMF->d_keyB);
    if (gpuMF->d_slot) cudaFree(gpuMF->d_slot);
    if (gpuMF->d_rowStart) cudaFree(gpuMF->d_rowStart);
    if (gpuMF->d_rowEnd) cudaFree(gpuMF->d_rowEnd);
    if (gpuMF->d_blockHist) cudaFree(gpuMF->d_blockHist);
    if (gpuMF->d_blockStartLocal) cudaFree(gpuMF->d_blockStartLocal);
    if (gpuMF->d_blockCounter) cudaFree(gpuMF->d_blockCounter);
    if (gpuMF->d_groupTotal) cudaFree(gpuMF->d_groupTotal);
    if (gpuMF->d_groupPrefix) cudaFree(gpuMF->d_groupPrefix);
    if (gpuMF->d_digitTotal) cudaFree(gpuMF->d_digitTotal);
    if (gpuMF->d_digitStart) cudaFree(gpuMF->d_digitStart);
    if (gpuMF->d_cand) cudaFree(gpuMF->d_cand);
    if (gpuMF->d_seq) cudaFree(gpuMF->d_seq);
    if (gpuMF->d_nbSeq) cudaFree(gpuMF->d_nbSeq);
    if (gpuMF->d_blockMeta) cudaFree(gpuMF->d_blockMeta);
    if (gpuMF->h_cand[0]) cudaFreeHost(gpuMF->h_cand[0]);
    if (gpuMF->h_cand[1]) cudaFreeHost(gpuMF->h_cand[1]);
    if (gpuMF->h_staging[0]) cudaFreeHost(gpuMF->h_staging[0]);
    if (gpuMF->h_staging[1]) cudaFreeHost(gpuMF->h_staging[1]);
    if (gpuMF->h_nbSeq[0]) cudaFreeHost(gpuMF->h_nbSeq[0]);
    if (gpuMF->h_nbSeq[1]) cudaFreeHost(gpuMF->h_nbSeq[1]);
    if (gpuMF->h_seq[0]) cudaFreeHost(gpuMF->h_seq[0]);
    if (gpuMF->h_seq[1]) cudaFreeHost(gpuMF->h_seq[1]);
    if (gpuMF->stream) cudaStreamDestroy(gpuMF->stream);
    if (gpuMF->copyStream) cudaStreamDestroy(gpuMF->copyStream);
    for (int b = 0; b < 2; b++) {
        if (gpuMF->evH2d[b]) cudaEventDestroy(gpuMF->evH2d[b]);
        if (gpuMF->evCompute[b]) cudaEventDestroy(gpuMF->evCompute[b]);
        if (gpuMF->evD2H[b]) cudaEventDestroy(gpuMF->evD2H[b]);
        if (gpuMF->evNbSeq[b]) cudaEventDestroy(gpuMF->evNbSeq[b]);
    }
    free(gpuMF);
    if (g_gpuRefCount > 0) g_gpuRefCount--;
    return 0;
}

/* ======================================================================== */
/* Host: batch pipeline                                                      */
/* ======================================================================== */

/* Reset the window/table state for a new frame. Must be called on a fresh
 * frame before any block is added. The table zeroing is issued on the compute
 * stream so it orders before the first batch's search. */
size_t ZSTD_gpuMatchFinder_reset(ZSTD_GpuMatchFinder* gpuMF)
{
    if (!gpuMF) return 0;
    if (cudaMemsetAsync(gpuMF->d_table, 0, (size_t)(1u << gpuMF->hashLog) * 4, gpuMF->stream) != cudaSuccess)
        return ERROR(GPU_cudaMemcpyFailed);
    gpuMF->winBase = 0;
    gpuMF->winUsed = 0;
    gpuMF->framePos = 0;
    gpuMF->batchCount = 0;
    gpuMF->batchBuf = 0;
    gpuMF->batchStagingOffset[0] = 0;
    gpuMF->batchSeqsCopied[0] = gpuMF->batchSeqsCopied[1] = 0;
    gpuMF->batchIdx = 0;
    return 0;
}

/* Stage a block into the current batch. Returns the in-batch index (0-based),
 * or a negative value on error. The caller submits the batch (via
 * ZSTD_gpuBatch_submit) when it is full or at the end of the input. */
int ZSTD_gpuBatch_addBlock(ZSTD_GpuMatchFinder* gpuMF, void const* src, size_t srcSize)
{
    if (!gpuMF || !src) return -1;
    if (srcSize > ZSTD_GPU_BLOCKSIZE) return -1;
    if (gpuMF->batchCount >= gpuMF->B) return -1;
    {
        unsigned const inBatch = gpuMF->batchCount;
        size_t const off = gpuMF->batchStagingOffset[inBatch];
        memcpy(gpuMF->h_staging[gpuMF->batchBuf] + off, src, srcSize);
        gpuMF->batchSize[inBatch] = (unsigned)srcSize;
        gpuMF->batchStagingOffset[inBatch + 1] = off + srcSize;
        gpuMF->batchCount++;
        return (int)inBatch;
    }
}

/* Submit the current batch to the GPU (asynchronous). Issues:
 *   copy stream  : [window memmove] + H2D append
 *   compute      : wait -> fillKeys -> radix -> slot -> rowBounds -> search
 *                  -> insert
 *   copy stream  : wait -> D2H of the per-position candidates
 * The batch state is reset so the next batch can be staged. */
size_t ZSTD_gpuBatch_submit(ZSTD_GpuMatchFinder* gpuMF)
{
    if (!gpuMF) return ERROR(GPU_invalidParameter);
    {
        unsigned const buf = gpuMF->batchBuf;
        unsigned const count = gpuMF->batchCount;
        if (count == 0) return 0;
        size_t const totalBytes = gpuMF->batchStagingOffset[count];

        /* Build the per-block metadata (absolute start, size) on the host and
         * copy it to the device. */
        static unsigned blockMetaHost[ZSTD_GPU_MAX_BATCH * 2];
        {
            unsigned abs = gpuMF->framePos;
            for (unsigned i = 0; i < count; i++) {
                blockMetaHost[i*2] = abs;
                blockMetaHost[i*2+1] = gpuMF->batchSize[i];
                abs += gpuMF->batchSize[i];
            }
        }
        if (cudaMemcpyAsync(gpuMF->d_blockMeta, blockMetaHost, count * 8,
                            cudaMemcpyHostToDevice, gpuMF->copyStream) != cudaSuccess)
            return ERROR(GPU_cudaMemcpyFailed);

        /* Window management: if appending this batch would overflow the window
         * buffer (windowSize + one block), slide the window forward first. The
         * slide is an overlapping device copy, so it goes through a temp buffer
         * (cudaMemcpyAsync is undefined for overlapping src/dst). */
        {
            size_t const needed = gpuMF->winUsed + totalBytes;
            size_t const cap = (size_t)gpuMF->windowSize + (size_t)gpuMF->B * ZSTD_GPU_BLOCKSIZE;
            if (needed > cap) {
                size_t const shift = needed - cap;
                if (cudaMemcpyAsync(gpuMF->d_windowTmp, gpuMF->d_window, gpuMF->winUsed,
                                    cudaMemcpyDeviceToDevice, gpuMF->copyStream) != cudaSuccess)
                    return ERROR(GPU_cudaMemcpyFailed);
                if (cudaMemcpyAsync(gpuMF->d_window, (unsigned char*)gpuMF->d_windowTmp + shift,
                                    gpuMF->winUsed - shift, cudaMemcpyDeviceToDevice,
                                    gpuMF->copyStream) != cudaSuccess)
                    return ERROR(GPU_cudaMemcpyFailed);
                gpuMF->winBase += (unsigned)shift;
                gpuMF->winUsed -= shift;
            }
        }

        /* H2D: append the batch bytes to the device window. */
        if (cudaMemcpyAsync((unsigned char*)gpuMF->d_window + gpuMF->winUsed,
                            gpuMF->h_staging[buf], totalBytes,
                            cudaMemcpyHostToDevice, gpuMF->copyStream) != cudaSuccess)
            return ERROR(GPU_cudaMemcpyFailed);
        gpuMF->winUsed += totalBytes;
        if (cudaEventRecord(gpuMF->evH2d[buf], gpuMF->copyStream) != cudaSuccess)
            return ERROR(GPU_cudaMemcpyFailed);

        /* Compute: wait for the H2D, then run the pipeline. */
        if (cudaStreamWaitEvent(gpuMF->stream, gpuMF->evH2d[buf], 0) != cudaSuccess)
            return ERROR(GPU_cudaMemcpyFailed);
        {
            unsigned totalPos = 0;
            for (unsigned i = 0; i < count; i++) {
                unsigned const sz = gpuMF->batchSize[i];
                totalPos += (sz >= 8) ? (sz - 7) : 0;
            }
            unsigned const nBlocks = (totalPos + ZSTD_GPU_THREADS - 1) / ZSTD_GPU_THREADS;
            unsigned const nGroups = (nBlocks + 255) / 256;
            unsigned const grid = (totalPos + ZSTD_GPU_THREADS - 1) / ZSTD_GPU_THREADS;

            /* Zero the scatter counters for the blocks we will use. */
            if (cudaMemsetAsync(gpuMF->d_blockCounter, 0, (size_t)nBlocks * 256 * 4, gpuMF->stream) != cudaSuccess)
                return ERROR(GPU_cudaMemcpyFailed);

            gpuFillKeys<<<grid, ZSTD_GPU_THREADS, 0, gpuMF->stream>>>(
                (unsigned char*)gpuMF->d_window, gpuMF->winBase,
                (unsigned*)gpuMF->d_blockMeta, count,
                gpuMF->hashLog, gpuMF->mls, gpuMF->posBits,
                (unsigned long long*)gpuMF->d_keyA, totalPos);

            /* Radix passes over the hash bits (the low posBits bits are already
             * in order). */
            {
                unsigned long long* cur = (unsigned long long*)gpuMF->d_keyA;
                unsigned long long* nxt = (unsigned long long*)gpuMF->d_keyB;
                for (unsigned pass = 0; pass < gpuMF->nPasses; pass++) {
                    unsigned const bitOffset = gpuMF->posBits + pass * 8;
                    gpuRadixHist<<<nBlocks, ZSTD_GPU_THREADS, 0, gpuMF->stream>>>(
                        cur, bitOffset, totalPos, (unsigned*)gpuMF->d_blockHist);
                    gpuRadixScanL1<<<nGroups, ZSTD_GPU_THREADS, 0, gpuMF->stream>>>(
                        (unsigned*)gpuMF->d_blockHist, nBlocks,
                        (unsigned*)gpuMF->d_blockStartLocal, (unsigned*)gpuMF->d_groupTotal);
                    gpuRadixScanL2<<<1, ZSTD_GPU_THREADS, 0, gpuMF->stream>>>(
                        (unsigned*)gpuMF->d_groupTotal, nGroups,
                        (unsigned*)gpuMF->d_digitTotal, (unsigned*)gpuMF->d_digitStart,
                        (unsigned*)gpuMF->d_groupPrefix);
                    /* Re-zero the counters for this pass (the previous pass used them). */
                    if (cudaMemsetAsync(gpuMF->d_blockCounter, 0, (size_t)nBlocks * 256 * 4, gpuMF->stream) != cudaSuccess)
                        return ERROR(GPU_cudaMemcpyFailed);
                    gpuRadixScatter<<<nBlocks, ZSTD_GPU_THREADS, 0, gpuMF->stream>>>(
                        cur, bitOffset, totalPos,
                        (unsigned*)gpuMF->d_digitStart, (unsigned*)gpuMF->d_groupPrefix,
                        (unsigned*)gpuMF->d_blockStartLocal, (unsigned*)gpuMF->d_blockCounter,
                        nxt);
                    /* Swap. */
                    { unsigned long long* tmp = cur; cur = nxt; nxt = tmp; }
                }
                /* After an even number of passes the result is in d_keyA; after
                 * an odd number it is in d_keyB. Copy to d_keyA if needed. */
                if (cur != (unsigned long long*)gpuMF->d_keyA) {
                    if (cudaMemcpyAsync(gpuMF->d_keyA, cur, (size_t)totalPos * 8,
                                        cudaMemcpyDeviceToDevice, gpuMF->stream) != cudaSuccess)
                        return ERROR(GPU_cudaMemcpyFailed);
                }
            }

            gpuRecordSlot<<<grid, ZSTD_GPU_THREADS, 0, gpuMF->stream>>>(
                (unsigned long long*)gpuMF->d_keyA, gpuMF->posBits, totalPos,
                (unsigned*)gpuMF->d_slot);

            {
                unsigned const rgrid = ((1u << gpuMF->hashLog) + ZSTD_GPU_THREADS - 1) / ZSTD_GPU_THREADS;
                gpuRowBoundsInit<<<rgrid, ZSTD_GPU_THREADS, 0, gpuMF->stream>>>(
                    (unsigned*)gpuMF->d_rowStart, (unsigned*)gpuMF->d_rowEnd,
                    gpuMF->hashLog, totalPos);
                gpuRowBounds<<<grid, ZSTD_GPU_THREADS, 0, gpuMF->stream>>>(
                    (unsigned long long*)gpuMF->d_keyA, gpuMF->posBits, totalPos,
                    (unsigned*)gpuMF->d_rowStart, (unsigned*)gpuMF->d_rowEnd);
            }

            /* Clear the batch's candidate slots before the search. The search
             * only writes positions that yield a valid candidate, and the
             * greedy treats any non-zero offset as a match : stale data from a
             * previous batch must not leak through as a phantom match. */
            if (cudaMemsetAsync(gpuMF->d_cand, 0, (size_t)count * ZSTD_GPU_BLOCKSIZE * 8, gpuMF->stream) != cudaSuccess)
                return ERROR(GPU_cudaMemcpyFailed);

            gpuSearch<<<grid, ZSTD_GPU_THREADS, 0, gpuMF->stream>>>(
                (unsigned char*)gpuMF->d_window, gpuMF->winBase,
                (unsigned*)gpuMF->d_blockMeta, count,
                (unsigned long long*)gpuMF->d_keyA, (unsigned*)gpuMF->d_slot,
                (unsigned*)gpuMF->d_table,
                gpuMF->hashLog, gpuMF->mls, gpuMF->posBits,
                gpuMF->windowSize, gpuMF->minMatch, totalPos,
                (unsigned*)gpuMF->d_cand);

            gpuInsert<<<grid, ZSTD_GPU_THREADS, 0, gpuMF->stream>>>(
                (unsigned char*)gpuMF->d_window, gpuMF->winBase,
                (unsigned*)gpuMF->d_blockMeta, count,
                gpuMF->hashLog, gpuMF->mls, totalPos,
                (unsigned*)gpuMF->d_table);

            gpuGreedy<<<count, 1, 0, gpuMF->stream>>>(
                (unsigned char*)gpuMF->d_window, gpuMF->winBase,
                (unsigned*)gpuMF->d_blockMeta, count,
                (unsigned*)gpuMF->d_cand,
                (ZSTD_GpuSequence*)gpuMF->d_seq,
                (unsigned*)gpuMF->d_nbSeq);
        }
        if (cudaEventRecord(gpuMF->evCompute[buf], gpuMF->stream) != cudaSuccess)
            return ERROR(GPU_cudaMemcpyFailed);

        /* D2H: the per-block sequence counts (small). The sequence data itself
         * is copied in waitBlock, once the host knows the counts. */
        if (cudaStreamWaitEvent(gpuMF->copyStream, gpuMF->evCompute[buf], 0) != cudaSuccess)
            return ERROR(GPU_cudaMemcpyFailed);
        if (cudaMemcpyAsync(gpuMF->h_nbSeq[buf], gpuMF->d_nbSeq,
                            (size_t)count * 4,
                            cudaMemcpyDeviceToHost, gpuMF->copyStream) != cudaSuccess)
            return ERROR(GPU_cudaMemcpyFailed);
        if (cudaEventRecord(gpuMF->evNbSeq[buf], gpuMF->copyStream) != cudaSuccess)
            return ERROR(GPU_cudaMemcpyFailed);

        /* Record the batch info for waitBlock. */
        gpuMF->batchInfo[buf].buf = buf;
        gpuMF->batchInfo[buf].count = count;
        gpuMF->batchInfo[buf].seqsCopied = 0;

        /* Advance the frame position and reset the batch state. */
        gpuMF->framePos += (unsigned)totalBytes;
        gpuMF->batchCount = 0;
        gpuMF->batchStagingOffset[0] = 0;
        gpuMF->batchBuf ^= 1;
        gpuMF->batchIdx++;
        return 0;
    }
}

/* Wait for the result of block `blockInBatch` (within batch `batchIdx`) and
 * copy its final sequences (position, offset, matchLength) into dst. The
 * host runs the greedy resolution (non-overlapping, in position order) over
 * the D2H'd candidates, extending each capped match to its true length.
 * Returns the number of sequences, or an error. */
size_t ZSTD_gpuBatch_waitBlock(ZSTD_GpuMatchFinder* gpuMF, unsigned batchIdx,
                               unsigned blockInBatch,
                               ZSTD_GpuSequence* dst, size_t dstCapacity,
                               void const* src, size_t srcSize)
{
    (void)src; (void)srcSize;   /* the device greedy uses the window, not src */
    if (!gpuMF || !dst) return ERROR(GPU_invalidParameter);
    {
        unsigned const slot = batchIdx % 2;
        ZSTD_GpuBatchInfo* info = &gpuMF->batchInfo[slot];
        if (blockInBatch >= info->count) return ERROR(GPU_invalidParameter);

        /* Wait for the nbSeq D2H (small) so the host knows the counts. */
        if (cudaEventSynchronize(gpuMF->evNbSeq[slot]) != cudaSuccess)
            return ERROR(GPU_cudaMemcpyFailed);

        /* On the first block of the batch, issue the sequence D2H for every
         * block in the batch (the host then reads them from the pinned mirror). */
        if (!info->seqsCopied) {
            for (unsigned i = 0; i < info->count; i++) {
                unsigned const nb = gpuMF->h_nbSeq[slot][i];
                if (nb > ZSTD_GPU_MAX_SEQS) return ERROR(GPU_allocationFailed);
                if (nb == 0) continue;
                if (cudaMemcpyAsync(gpuMF->h_seq[slot] + (size_t)i * ZSTD_GPU_MAX_SEQS,
                                    (ZSTD_GpuSequence*)gpuMF->d_seq + (size_t)i * ZSTD_GPU_MAX_SEQS,
                                    (size_t)nb * sizeof(ZSTD_GpuSequence),
                                    cudaMemcpyDeviceToHost, gpuMF->copyStream) != cudaSuccess)
                    return ERROR(GPU_cudaMemcpyFailed);
            }
            if (cudaEventRecord(gpuMF->evD2H[slot], gpuMF->copyStream) != cudaSuccess)
                return ERROR(GPU_cudaMemcpyFailed);
            info->seqsCopied = 1;
        }
        if (cudaEventSynchronize(gpuMF->evD2H[slot]) != cudaSuccess)
            return ERROR(GPU_cudaMemcpyFailed);

        {
            unsigned const nb = gpuMF->h_nbSeq[slot][blockInBatch];
            if (nb > dstCapacity) return ERROR(GPU_allocationFailed);
            ZSTD_GpuSequence const* const s = gpuMF->h_seq[slot] + (size_t)blockInBatch * ZSTD_GPU_MAX_SEQS;
            for (unsigned i = 0; i < nb; i++) dst[i] = s[i];
            return nb;
        }
    }
}

/* Synchronous single-block path (used by the non-frame block mode and as a
 * fallback): stage one block, submit, wait, return its sequences. */
size_t ZSTD_compressBlock_gpu(
    ZSTD_GpuMatchFinder* gpuMF,
    ZSTD_Sequence* dst,
    size_t dstCapacity,
    void const* src,
    size_t srcSize,
    unsigned rep[ZSTD_REP_NUM],
    unsigned repOut[ZSTD_REP_NUM])
{
    (void)rep; (void)repOut;   /* the GPU path never emits repcodes */
    if (!gpuMF || !src) return ERROR(GPU_invalidParameter);
    if (srcSize < 4) return 0;
    {
        int const inBatch = ZSTD_gpuBatch_addBlock(gpuMF, src, srcSize);
        if (inBatch < 0) return ERROR(GPU_allocationFailed);
        {
            size_t const err = ZSTD_gpuBatch_submit(gpuMF);
            if (ZSTD_isError(err)) return err;
            /* The block is inBatch of the just-submitted batch (batchIdx-1). */
            unsigned const batchIdx = gpuMF->batchIdx - 1;
            return ZSTD_gpuBatch_waitBlock(gpuMF, batchIdx, (unsigned)inBatch,
                                           (ZSTD_GpuSequence*)dst, dstCapacity, src, srcSize);
        }
    }
}
