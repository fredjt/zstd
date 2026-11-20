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
 *                    search (candidate = max(in-batch row
 *                    predecessor, pre-batch table entry)) -> insert
 *                    (atomicMax table update for the next batch)
 *      copy stream : wait -> D2H of the per-position candidates
 *    Pinned staging is double-buffered so the H2D of batch k+1 overlaps the
 *    compute of batch k (G10). One sync per batch, not per block (G11).
 *
 *  - The search uses the CPU's exact fingerprint hash (minMatch-dependent),
 *    an 8-byte prefilter and a window-age check before the per-position count
 *    (G12). Greedy sequence resolution (non-overlapping, in position order)
 *    runs on the device (G13) : a sequential walk per block that jumps past
 *    each emitted match, with the match extension unrolled for memory-level
 *    parallelism (G18.1).
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
        /* Prefix over the per-group totals for this digit : group g starts at
         * the sum of the digit's counts in all earlier groups. */
        unsigned running = 0;
        for (unsigned g = 0; g < nGroups; g++) {
            d_groupPrefix[g * 256 + d] = running;
            running += d_groupTotal[g * 256 + d];
        }
    }
}

/* Stable scatter: place each key into d_keysB at its sorted position.
 * dst = digitStart[d] + groupPrefix[g][d] + blockStartLocal[cb][d] + localRank.
 * The localRank is the rank of this key within (cb, d) *in input order*, so
 * the LSD radix sort is stable and the low (position) bits stay ordered
 * within each hash row. Each thread owns exactly one input element in
 * thread-index (input) order, so the rank is the number of lower-indexed
 * threads in this block that carry the same digit. */
__global__ void gpuRadixScatter(const unsigned long long* __restrict__ d_keys,
                                unsigned bitOffset, unsigned totalPos,
                                const unsigned* __restrict__ d_digitStart,
                                const unsigned* __restrict__ d_groupPrefix,
                                const unsigned* __restrict__ d_blockStartLocal,
                                unsigned long long* __restrict__ d_keysB)
{
    __shared__ unsigned s_digit[256];
    unsigned const t = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long key = 0;
    unsigned dd = 0xFFu;   /* sentinel: never matches a real digit */
    if (t < totalPos) {
        key = d_keys[t];
        dd = (unsigned)((key >> bitOffset) & 255);
    }
    s_digit[threadIdx.x] = dd;
    __syncthreads();
    if (t < totalPos) {
        unsigned localRank = 0;
        for (unsigned j = 0; j < threadIdx.x; j++) {
            if (s_digit[j] == dd) localRank++;
        }
        unsigned const g = blockIdx.x / 256;
        unsigned const dst = d_digitStart[dd] + d_groupPrefix[g * 256 + dd]
                           + d_blockStartLocal[blockIdx.x * 256 + dd] + localRank;
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
                          unsigned* __restrict__ d_cand,
                          unsigned* __restrict__ d_nbCand)
{
    /* G18.4 : the candidate count is reduced per CUDA block (a shared counter
     * plus one global atomic per block) instead of one global atomic per
     * candidate position. On periodic data every position is a candidate, so
     * the per-position version serialized ~131K global atomics on one address
     * per 128 KB block. No early returns : every thread reaches the final
     * __syncthreads. */
    __shared__ unsigned s_count[ZSTD_GPU_MAX_BATCH];
    if (threadIdx.x < B) s_count[threadIdx.x] = 0;
    __syncthreads();
    unsigned const t = blockIdx.x * blockDim.x + threadIdx.x;
    int isCand = 0;
    unsigned i = 0;
    if (t < totalPos) {
        unsigned a, pib;
        gpuBatchLocate(d_blockMeta, B, t, i, a, pib);
        unsigned char const* const p = d_window + (a - winBase);
        unsigned const h = gpuHash(p, hashLog, mls);

        /* In-batch predecessor: the element just before t in the sorted array,
         * if it has the same hash. Position 0 is a valid absolute position, so
         * track validity separately. */
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

        int pass = hasQ;
        if (pass) {
            /* Window-age check (G12.2): skip candidates older than the window
             * before touching the data. */
            pass = (a - qc <= windowSize);
        }
        if (pass) {
            unsigned char const* const qp = d_window + (qc - winBase);
            /* 8-byte prefilter (G12.1): two 64-bit loads at the candidate head.
             * Valid positions always have 8 bytes left in the block (nValid =
             * size-7), so the read stays inside the block. For the GPU levels
             * minMatch <= 8, so a passing prefilter means the candidate is at
             * least minMatch long; the capped length is ZSTD_GPU_MATCH_CAP and
             * the greedy extends it. */
            (void)minMatch;
            pass = (gpuRead64(p) == gpuRead64(qp));
        }
        if (pass) {
            unsigned* const c = d_cand + (i * ZSTD_GPU_BLOCKSIZE + pib) * 2;
            c[0] = a - qc;                    /* offset */
            c[1] = ZSTD_GPU_MATCH_CAP;        /* capped length (the greedy extends it) */
            isCand = 1;
        }
    }
    if (isCand) atomicAdd(&s_count[i], 1);   /* per-CUDA-block reduction */
    __syncthreads();
    for (unsigned k = threadIdx.x; k < B; k += blockDim.x) {
        if (s_count[k]) atomicAdd(&d_nbCand[k], s_count[k]);   /* one global atomic per block */
    }
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
/* Device greedy (G18.1) : one warp (32 threads) per block walks the
 * positions in order and emits non-overlapping matches, jumping past each
 * emitted match (pib += len) instead of stepping through the covered
 * positions. Thread 0 drives the sequential walk (finding the match-starts);
 * the whole warp cooperatively extends each match : 32 lanes x 8 bytes =
 * 256 bytes per round, so a whole-block match is ~512 rounds of coalesced
 * 64-bit loads instead of 16K dependent single-thread steps. Before G18.1
 * the extension was a single-thread byte/chunk loop, so a whole-block match
 * cost ~1.9 ms per 128 KB block (~90 % of the batch).
 *
 * A first-mismatch scan (not a binary search) is required because the
 * per-chunk equality is not monotonic : the data may match, mismatch, then
 * match again. The match is capped at the block end, exactly like the CPU's
 * ZSTD_count (the source always has the bytes : qc + maxLen < the block end
 * < the window end).
 * ======================================================================== */
__global__ void gpuGreedy(const unsigned char* __restrict__ d_window, unsigned winBase,
                          const unsigned* __restrict__ d_blockMeta, unsigned B,
                          const unsigned* __restrict__ d_cand,
                          const unsigned* __restrict__ d_nbCand,
                          ZSTD_GpuSequence* __restrict__ d_seq,
                          unsigned* __restrict__ d_nbSeq)
{
    unsigned const i = blockIdx.x;   /* frame block */
    if (i >= B) return;
    __shared__ unsigned s_pib, s_off, s_len, s_state;   /* state : 0 = walk, 1 = done */
    unsigned const size = d_blockMeta[i*2+1];
    unsigned const absStart = d_blockMeta[i*2];
    if (threadIdx.x == 0) {
        s_pib = 0; s_off = 0; s_len = 0;
        s_state = (d_nbCand[i] == 0) ? 1 : 0;
    }
    __syncthreads();
    if (s_state == 1) { if (threadIdx.x == 0) d_nbSeq[i] = 0; return; }

    unsigned nbSeq = 0;
    for (;;) {
        /* Phase A : thread 0 finds the next match-start (or marks done). */
        if (threadIdx.x == 0) {
            unsigned pib = s_pib;
            unsigned off = 0;
            while (pib + 1 < size) {
                off = d_cand[(i * ZSTD_GPU_BLOCKSIZE + pib) * 2];
                if (off != 0) break;
                pib++;
            }
            if (pib + 1 >= size) { s_state = 1; }
            else { s_pib = pib; s_off = off; }
        }
        __syncthreads();
        if (s_state == 1) break;
        unsigned const pib = s_pib;
        unsigned const off = s_off;
        /* Phase B : the warp cooperatively extends the match at pib.
         * The first 8 bytes are known equal (the search's prefilter), so the
         * extension covers [8, maxLen). Round r covers [8+256r, 8+256r+256);
         * lane L checks the 8 bytes at [8+256r+8L, +8). */
        unsigned const a = absStart + pib;
        unsigned const maxLen = size - pib;
        const unsigned char* const base = d_window + (a - winBase);
        const unsigned char* const srcp = d_window + (a - off - winBase);
        unsigned len = ZSTD_GPU_MATCH_CAP;   /* 8 : the prefilter already matched */
        while (len + 256 <= maxLen) {
            unsigned const byteOff = len + threadIdx.x * 8;
            unsigned long long const b = gpuRead64(base + byteOff);
            unsigned long long const s = gpuRead64(srcp + byteOff);
            unsigned const bal = __ballot_sync(0xFFFFFFFFu, b != s);
            if (bal == 0) { len += 256; continue; }   /* all 32 chunks equal */
            unsigned const lane = (unsigned)__ffs(bal) - 1;   /* first mismatching lane */
            if (threadIdx.x == lane) {
                unsigned local = len + lane * 8;
                while (local < maxLen && base[local] == srcp[local]) local++;
                s_len = local;
            }
            __syncthreads();
            len = s_len;
            break;
        }
        /* Byte tail : [len, maxLen) is < 264 bytes ; thread 0 finishes it. */
        if (threadIdx.x == 0) {
            while (len < maxLen && base[len] == srcp[len]) len++;
            s_len = len;
        }
        __syncthreads();
        len = s_len;
        /* Phase C : thread 0 emits the sequence and advances the walk. */
        if (threadIdx.x == 0) {
            ZSTD_GpuSequence* const s = d_seq + (i * ZSTD_GPU_MAX_SEQS + nbSeq);
            s->position = pib;
            s->offset = off;
            s->matchLength = len;
            nbSeq++;
            s_pib = pib + len;   /* jump past the emitted match */
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) d_nbSeq[i] = nbSeq;
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
        bytes += (size_t)B * 512 * 256 * 4 * 2;                            /* hist/startLocal */
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
    /* POC: at most one GPU match finder per process. The claim must be atomic:
     * in multithreaded compression several worker contexts are initialized
     * concurrently, and a plain check-then-set lets more than one of them
     * through, creating several match finders whose independent windows and
     * streams produce corrupted frames. */
    {
        int expected = 0;
        if (!__atomic_compare_exchange_n(&g_gpuRefCount, &expected, 1, 0,
                                         __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST))
            return ERROR(GPU_allocationFailed);
    }

    ZSTD_GpuMatchFinder* mf = NULL;
    {
        mf = (ZSTD_GpuMatchFinder*)calloc(1, sizeof(*mf));
        if (!mf) {
            __atomic_fetch_sub(&g_gpuRefCount, 1, __ATOMIC_SEQ_CST);  /* release the claim */
            return ERROR(GPU_allocationFailed);
        }

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
        /* The window buffer must hold the window (windowSize) plus the bytes of
         * the batches the GPU may still be computing. The chunk pipeline keeps
         * at most 2 batches in flight (the double-buffered staging limit), so
         * the headroom is 2 full batches (G19). */
        {
            size_t const wbuf = (size_t)mf->windowSize + (size_t)2 * mf->B * ZSTD_GPU_BLOCKSIZE;
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
        {
            size_t const hbytes = (size_t)mf->nBlocksMax * 256 * 4;
            if (cudaMalloc(&mf->d_blockHist, hbytes) != cudaSuccess) goto fail;
            if (cudaMalloc(&mf->d_blockStartLocal, hbytes) != cudaSuccess) goto fail;
            size_t const gbytes = (size_t)((mf->nBlocksMax + 255) / 256) * 256 * 4;
            if (cudaMalloc(&mf->d_groupTotal, gbytes) != cudaSuccess) goto fail;
            if (cudaMalloc(&mf->d_groupPrefix, gbytes) != cudaSuccess) goto fail;
            if (cudaMalloc(&mf->d_digitTotal, 256 * 4) != cudaSuccess) goto fail;
            if (cudaMalloc(&mf->d_digitStart, 256 * 4) != cudaSuccess) goto fail;
        }
        if (cudaMalloc(&mf->d_cand, (size_t)mf->B * ZSTD_GPU_BLOCKSIZE * 8) != cudaSuccess) goto fail;
        if (cudaMalloc(&mf->d_nbCand, (size_t)mf->B * 4) != cudaSuccess) goto fail;
        for (int b = 0; b < 2; b++) {
            if (cudaMalloc(&mf->d_seq[b], (size_t)mf->B * ZSTD_GPU_MAX_SEQS * 12) != cudaSuccess) goto fail;
            if (cudaMalloc(&mf->d_nbSeq[b], (size_t)mf->B * 4) != cudaSuccess) goto fail;
        }
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
        {
            const char* const pe = getenv("ZSTD_GPU_PROFILE");
            if (pe && pe[0] && pe[0] != '0') {
                mf->profile = 1;
                for (int p = 0; p < 8; p++) {
                    if (cudaEventCreate(&mf->profEv[2*p]) != cudaSuccess) goto fail;
                    if (cudaEventCreate(&mf->profEv[2*p+1]) != cudaSuccess) goto fail;
                }
            }
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

        *gpuMF = mf;   /* g_gpuRefCount was claimed atomically above */
        return 0;
    }
fail:
    ZSTD_freeGpuMatchFinder(mf);
    return ERROR(GPU_allocationFailed);
}

/* G17: print the per-phase breakdown (averaged over all batches). */
static void gpuProfReport(const ZSTD_GpuMatchFinder* mf)
{
    if (!mf->profile || mf->phaseCount == 0) return;
    static const char* const names[8] = {
        "H2D", "fillKeys", "radix", "recordSlot",
        "search", "insert", "greedy", "D2H"};
    double total = 0;
    for (int p = 0; p < 8; p++) total += mf->phaseMs[p];
    fprintf(stderr, "[ZSTD_GPU_PROFILE] per 128KB block (avg over %u batches)\n", (unsigned)mf->phaseCount);
    for (int p = 0; p < 8; p++) {
        double avg = mf->phaseMs[p] / (double)mf->phaseCount;
        double pct = (total > 0) ? 100.0 * avg / (total / (double)mf->phaseCount) : 0.0;
        fprintf(stderr, "  %-11s %9.4f ms  %5.1f%%\n", names[p], avg, pct);
    }
    fprintf(stderr, "  %-11s %9.4f ms\n", "total", total / (double)mf->phaseCount);
}

size_t ZSTD_freeGpuMatchFinder(ZSTD_GpuMatchFinder* gpuMF)
{
    if (!gpuMF) return 0;
    gpuProfReport(gpuMF);
    /* Drain any in-flight work before freeing the buffers it touches
     * (the runtime fallback path can free the MF mid-frame). */
    if (gpuMF->stream) cudaStreamSynchronize(gpuMF->stream);
    if (gpuMF->copyStream) cudaStreamSynchronize(gpuMF->copyStream);
    if (gpuMF->d_window) cudaFree(gpuMF->d_window);
    if (gpuMF->d_windowTmp) cudaFree(gpuMF->d_windowTmp);
    if (gpuMF->d_table) cudaFree(gpuMF->d_table);
    if (gpuMF->d_keyA) cudaFree(gpuMF->d_keyA);
    if (gpuMF->d_keyB) cudaFree(gpuMF->d_keyB);
    if (gpuMF->d_slot) cudaFree(gpuMF->d_slot);
    if (gpuMF->d_blockHist) cudaFree(gpuMF->d_blockHist);
    if (gpuMF->d_blockStartLocal) cudaFree(gpuMF->d_blockStartLocal);
    if (gpuMF->d_groupTotal) cudaFree(gpuMF->d_groupTotal);
    if (gpuMF->d_groupPrefix) cudaFree(gpuMF->d_groupPrefix);
    if (gpuMF->d_digitTotal) cudaFree(gpuMF->d_digitTotal);
    if (gpuMF->d_digitStart) cudaFree(gpuMF->d_digitStart);
    if (gpuMF->d_cand) cudaFree(gpuMF->d_cand);
    if (gpuMF->d_nbCand) cudaFree(gpuMF->d_nbCand);
    for (int b = 0; b < 2; b++) {
        if (gpuMF->d_seq[b]) cudaFree(gpuMF->d_seq[b]);
        if (gpuMF->d_nbSeq[b]) cudaFree(gpuMF->d_nbSeq[b]);
    }
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
    if (gpuMF->profile) for (int p = 0; p < 8; p++) {
        if (gpuMF->profEv[2*p]) cudaEventDestroy(gpuMF->profEv[2*p]);
        if (gpuMF->profEv[2*p+1]) cudaEventDestroy(gpuMF->profEv[2*p+1]);
    }
    free(gpuMF);
    if (g_gpuRefCount > 0) __atomic_fetch_sub(&g_gpuRefCount, 1, __ATOMIC_SEQ_CST);
    return 0;
}

/* Reset the match finder to its initial (empty) state, keeping the device
 * buffers. Called when the owning context is re-initialized - in particular
 * for every job in multithreaded compression, where the context's own window
 * is reset per job. Without this the GPU window would accumulate the previous
 * job's data (with a hole where the other workers' jobs were), the frame
 * position would be wrong, and stale hash-table entries would produce invalid
 * offsets - i.e. corrupted frames. */
size_t ZSTD_resetGpuMatchFinder(ZSTD_GpuMatchFinder* gpuMF)
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
    gpuMF->batchIdx = 0;
    gpuMF->batchInfo[0].seqsCopied = 0;
    gpuMF->batchInfo[1].seqsCopied = 0;
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
    /* G19 chunk pipeline state (the global block index restarts at 0). */
    gpuMF->chunkSrc = NULL;
    gpuMF->chunkSize = 0;
    gpuMF->chunkStartBlock = 0;
    gpuMF->chunkEndBlock = 0;
    gpuMF->nextWait = 0;
    gpuMF->curBatchStart = 0;
    gpuMF->histIdx = 0;
    gpuMF->histBatchIdx[0] = gpuMF->histBatchIdx[1] = 0;
    gpuMF->histStart[0] = gpuMF->histStart[1] = 0;
    gpuMF->histCount[0] = gpuMF->histCount[1] = 0;
    gpuMF->inFlight = 0;
    gpuMF->chunkMode = 0;
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
        if (getenv("ZSTD_GPU_DEBUG_SEQS")) fprintf(stderr, "  add: inBatch=%u count=%u size=%zu buf=%u\n", inBatch, gpuMF->batchCount, srcSize, gpuMF->batchBuf);
        return (int)inBatch;
    }
}

/* Submit the current batch to the GPU (asynchronous). Issues:
 *   copy stream  : [window memmove] + H2D append
 *   compute      : wait -> fillKeys -> radix -> slot -> search -> insert
 *   copy stream  : wait -> D2H of the per-position candidates
 * The batch state is reset so the next batch can be staged. */
/* G17 per-phase profiling helpers (no-ops unless ZSTD_GPU_PROFILE). */
static void gpuProfRec(ZSTD_GpuMatchFinder* mf, int idx, cudaStream_t st)
{
    if (mf->profile) (void)cudaEventRecord(mf->profEv[idx], st);
}
static void gpuProfAccum(ZSTD_GpuMatchFinder* mf)
{
    if (!mf->profile) return;
    for (int p = 0; p < 8; p++) {
        float ms = 0.f;
        (void)cudaEventElapsedTime(&ms, mf->profEv[2*p], mf->profEv[2*p+1]);
        mf->phaseMs[p] += ms;
    }
    mf->phaseCount++;
}

size_t ZSTD_gpuBatch_submit(ZSTD_GpuMatchFinder* gpuMF)
{
    if (!gpuMF) return ERROR(GPU_invalidParameter);
    {
        unsigned const buf = gpuMF->batchBuf;
        unsigned const count = gpuMF->batchCount;
        if (getenv("ZSTD_GPU_DEBUG_SEQS")) fprintf(stderr, "  submit: count=%u buf=%u batchIdx=%u\n", count, buf, gpuMF->batchIdx);
        if (count == 0) return 0;
        size_t const totalBytes = gpuMF->batchStagingOffset[count];

        /* Build the per-block metadata (absolute start, size) on the host and
         * copy it to the device. */
        /* Double-buffered per batch parity : the H2D is asynchronous, so the
         * host must not overwrite a buffer whose copy is still in flight (the
         * chunk pipeline submits the next batch before this one is consumed).
         * With at most 2 batches in flight, parity is safe. */
        static unsigned blockMetaHost[2][ZSTD_GPU_MAX_BATCH * 2];
        {
            unsigned abs = gpuMF->framePos;
            for (unsigned i = 0; i < count; i++) {
                blockMetaHost[buf][i*2] = abs;
                blockMetaHost[buf][i*2+1] = gpuMF->batchSize[i];
                abs += gpuMF->batchSize[i];
            }
        }
        if (cudaMemcpyAsync(gpuMF->d_blockMeta, blockMetaHost[buf], count * 8,
                            cudaMemcpyHostToDevice, gpuMF->copyStream) != cudaSuccess)
            return ERROR(GPU_cudaMemcpyFailed);

        /* Window management: if appending this batch would overflow the window
         * buffer (windowSize + the in-flight batch bytes, 2 batches with the
         * chunk pipeline), slide the window forward first. The slide discards
         * only history older than the window (the in-flight bytes are always
         * the newest ones). The slide is an overlapping device copy, so it
         * goes through a temp buffer (cudaMemcpyAsync is undefined for
         * overlapping src/dst). */
        {
            size_t const needed = gpuMF->winUsed + totalBytes;
            size_t const cap = (size_t)gpuMF->windowSize + (size_t)2 * gpuMF->B * ZSTD_GPU_BLOCKSIZE;
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
        gpuProfRec(gpuMF, 0, gpuMF->copyStream);   /* H2D phase start */
        if (cudaMemcpyAsync((unsigned char*)gpuMF->d_window + gpuMF->winUsed,
                            gpuMF->h_staging[buf], totalBytes,
                            cudaMemcpyHostToDevice, gpuMF->copyStream) != cudaSuccess)
            return ERROR(GPU_cudaMemcpyFailed);
        gpuMF->winUsed += totalBytes;
        if (cudaEventRecord(gpuMF->evH2d[buf], gpuMF->copyStream) != cudaSuccess)
            return ERROR(GPU_cudaMemcpyFailed);
        gpuProfRec(gpuMF, 1, gpuMF->copyStream);   /* H2D phase end */

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

            gpuProfRec(gpuMF, 2, gpuMF->stream);
            gpuFillKeys<<<grid, ZSTD_GPU_THREADS, 0, gpuMF->stream>>>(
                (unsigned char*)gpuMF->d_window, gpuMF->winBase,
                (unsigned*)gpuMF->d_blockMeta, count,
                gpuMF->hashLog, gpuMF->mls, gpuMF->posBits,
                (unsigned long long*)gpuMF->d_keyA, totalPos);
            gpuProfRec(gpuMF, 3, gpuMF->stream);

            /* Radix passes over the hash bits (the low posBits bits are already
             * in order). */
            gpuProfRec(gpuMF, 4, gpuMF->stream);
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
                    gpuRadixScatter<<<nBlocks, ZSTD_GPU_THREADS, 0, gpuMF->stream>>>(
                        cur, bitOffset, totalPos,
                        (unsigned*)gpuMF->d_digitStart, (unsigned*)gpuMF->d_groupPrefix,
                        (unsigned*)gpuMF->d_blockStartLocal,
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

            gpuProfRec(gpuMF, 5, gpuMF->stream);
            gpuProfRec(gpuMF, 6, gpuMF->stream);   /* recordSlot start */
            gpuRecordSlot<<<grid, ZSTD_GPU_THREADS, 0, gpuMF->stream>>>(
                (unsigned long long*)gpuMF->d_keyA, gpuMF->posBits, totalPos,
                (unsigned*)gpuMF->d_slot);
            gpuProfRec(gpuMF, 7, gpuMF->stream);

            /* Clear the batch's candidate slots before the search. The search
             * only writes positions that yield a valid candidate, and the
             * greedy treats any non-zero offset as a match : stale data from a
             * previous batch must not leak through as a phantom match. */
            if (cudaMemsetAsync(gpuMF->d_cand, 0, (size_t)count * ZSTD_GPU_BLOCKSIZE * 8, gpuMF->stream) != cudaSuccess)
                return ERROR(GPU_cudaMemcpyFailed);
            if (cudaMemsetAsync(gpuMF->d_nbCand, 0, (size_t)count * 4, gpuMF->stream) != cudaSuccess)
                return ERROR(GPU_cudaMemcpyFailed);

            gpuProfRec(gpuMF, 8, gpuMF->stream);
            gpuSearch<<<grid, ZSTD_GPU_THREADS, 0, gpuMF->stream>>>(
                (unsigned char*)gpuMF->d_window, gpuMF->winBase,
                (unsigned*)gpuMF->d_blockMeta, count,
                (unsigned long long*)gpuMF->d_keyA, (unsigned*)gpuMF->d_slot,
                (unsigned*)gpuMF->d_table,
                gpuMF->hashLog, gpuMF->mls, gpuMF->posBits,
                gpuMF->windowSize, gpuMF->minMatch, totalPos,
                (unsigned*)gpuMF->d_cand, (unsigned*)gpuMF->d_nbCand);
            gpuProfRec(gpuMF, 9, gpuMF->stream);   /* search end */

            gpuProfRec(gpuMF, 10, gpuMF->stream);   /* insert start */
            gpuInsert<<<grid, ZSTD_GPU_THREADS, 0, gpuMF->stream>>>(
                (unsigned char*)gpuMF->d_window, gpuMF->winBase,
                (unsigned*)gpuMF->d_blockMeta, count,
                gpuMF->hashLog, gpuMF->mls, totalPos,
                (unsigned*)gpuMF->d_table);
            gpuProfRec(gpuMF, 11, gpuMF->stream);   /* insert end */

            gpuProfRec(gpuMF, 12, gpuMF->stream);
            gpuGreedy<<<count, 32, 0, gpuMF->stream>>>(
                (unsigned char*)gpuMF->d_window, gpuMF->winBase,
                (unsigned*)gpuMF->d_blockMeta, count,
                (unsigned*)gpuMF->d_cand,
                (unsigned*)gpuMF->d_nbCand,
                (ZSTD_GpuSequence*)gpuMF->d_seq[buf],
                (unsigned*)gpuMF->d_nbSeq[buf]);
            gpuProfRec(gpuMF, 13, gpuMF->stream);
        }
        /* Catch kernel-launch failures (e.g. "no kernel image is available" when
         * the binary was built without this GPU's architecture). Without this
         * check the launches fail silently, every block comes back with zero
         * sequences, and the output degrades to all-literals with no warning.
         * The caller (ZSTD_buildSeqStore_gpu) turns the error into a one-time
         * warning plus the CPU match finder. */
        {
            cudaError_t const e = cudaGetLastError();
            if (e != cudaSuccess) return ERROR(GPU_kernelLaunchFailed);
        }
        if (cudaEventRecord(gpuMF->evCompute[buf], gpuMF->stream) != cudaSuccess)
            return ERROR(GPU_cudaMemcpyFailed);

        /* D2H: the per-block sequence counts (small). The sequence data itself
         * is copied in waitBlock, once the host knows the counts. */
        if (cudaStreamWaitEvent(gpuMF->copyStream, gpuMF->evCompute[buf], 0) != cudaSuccess)
            return ERROR(GPU_cudaMemcpyFailed);
        gpuProfRec(gpuMF, 14, gpuMF->copyStream);
        if (cudaMemcpyAsync(gpuMF->h_nbSeq[buf], gpuMF->d_nbSeq[buf],
                            (size_t)count * 4,
                            cudaMemcpyDeviceToHost, gpuMF->copyStream) != cudaSuccess)
            return ERROR(GPU_cudaMemcpyFailed);
        if (cudaEventRecord(gpuMF->evNbSeq[buf], gpuMF->copyStream) != cudaSuccess)
            return ERROR(GPU_cudaMemcpyFailed);
        gpuProfRec(gpuMF, 15, gpuMF->copyStream);

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
                                    (ZSTD_GpuSequence*)gpuMF->d_seq[slot] + (size_t)i * ZSTD_GPU_MAX_SEQS,
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

        /* G17: accumulate the per-phase times now that all of this batch's
         * GPU work is complete (the events were recorded during submit). Done
         * once per batch, on its last block, before the next submit overwrites
         * the events. */
        if (blockInBatch + 1 >= info->count) gpuProfAccum(gpuMF);

        {
            unsigned const nb = gpuMF->h_nbSeq[slot][blockInBatch];
            if (nb > dstCapacity) return ERROR(GPU_allocationFailed);
            ZSTD_GpuSequence const* const s = gpuMF->h_seq[slot] + (size_t)blockInBatch * ZSTD_GPU_MAX_SEQS;
            for (unsigned i = 0; i < nb; i++) dst[i] = s[i];
            if (getenv("ZSTD_GPU_DEBUG_SEQS")) {
                static unsigned dbgBlk = 0;
                fprintf(stderr, "blk %u (batch %u inb %u) nb=%u\n", dbgBlk++, batchIdx, blockInBatch, nb);
                for (unsigned i = 0; i < nb; i++) fprintf(stderr, "  pos=%u off=%u len=%u\n", s[i].position, s[i].offset, s[i].matchLength);
            }
            return nb;
        }
    }
}

/* ======================================================================== */
/* Host-side chunk pipeline (G19)                                           */
/* ======================================================================== */

/* Submit the current (unsubmitted) batch and record it in the 2-slot
 * submitted-batch history (used by ZSTD_gpuChunk_processBlock to map a global
 * block index to its (batchIdx, in-batch index)). The double-buffering allows
 * at most 2 batches in flight, so a 2-slot history is sufficient (the batch
 * being waited on is always one of the last 2 submitted). */
static size_t gpuChunk_submitBatch(ZSTD_GpuMatchFinder* gpuMF)
{
    unsigned const start = gpuMF->curBatchStart;
    unsigned const count = gpuMF->batchCount;
    if (count == 0) return 0;
    size_t const err = ZSTD_gpuBatch_submit(gpuMF);
    if (ZSTD_isError(err)) return err;
    gpuMF->histBatchIdx[gpuMF->histIdx] = gpuMF->batchIdx - 1;  /* submit() incremented batchIdx */
    gpuMF->histStart[gpuMF->histIdx] = start;
    gpuMF->histCount[gpuMF->histIdx] = count;
    gpuMF->histIdx ^= 1;
    gpuMF->curBatchStart = start + count;
    gpuMF->inFlight++;
    return 0;
}

/* Begin a frame chunk for the chunk pipeline. The global block index of the
 * chunk's first block is the next block to stage (curBatchStart + batchCount),
 * which equals the previous chunk's end (the look-ahead is limited to the
 * current chunk, so it never stages past the chunk end). */
size_t ZSTD_gpuChunk_begin(ZSTD_GpuMatchFinder* gpuMF, void const* src, size_t srcSize)
{
    if (!gpuMF || !src) return ERROR(GPU_invalidParameter);
    gpuMF->chunkSrc = src;
    gpuMF->chunkSize = srcSize;
    gpuMF->chunkStartBlock = gpuMF->curBatchStart + gpuMF->batchCount;
    gpuMF->chunkEndBlock = gpuMF->chunkStartBlock
                         + (unsigned)((srcSize + ZSTD_GPU_BLOCKSIZE - 1) / ZSTD_GPU_BLOCKSIZE);
    return 0;
}

/* Stage the look-ahead: blocks from (curBatchStart + batchCount) up to
 * min(globalBlockIdx + 2*B, chunkEndBlock), submitting each batch as it fills.
 * The staging buffer is double-buffered (one unsubmitted batch per buffer),
 * and the host may only write batch k+2's buffer once batch k's H2D has
 * completed - which holds iff at most 2 batches are in flight. So a full batch
 * is submitted only while inFlight < 2; otherwise staging pauses until the
 * host encoder consumes another block (which frees a pipeline slot). The GPU
 * still works on batch k+1 while the host waits for / encodes batch k. */
static size_t gpuChunk_stageLookahead(ZSTD_GpuMatchFinder* gpuMF, unsigned globalBlockIdx)
{
    unsigned const upTo = (globalBlockIdx + 2 * gpuMF->B < gpuMF->chunkEndBlock)
                        ? (globalBlockIdx + 2 * gpuMF->B)
                        : gpuMF->chunkEndBlock;
    while (gpuMF->curBatchStart + gpuMF->batchCount < upTo) {
        if (gpuMF->batchCount >= gpuMF->B) {
            if (gpuMF->inFlight >= 2) break;   /* pipeline full; resume on the next block */
            FORWARD_IF_ERROR(gpuChunk_submitBatch(gpuMF), "GPU chunk batch submit failed");
        }
        {
            unsigned const b = gpuMF->curBatchStart + gpuMF->batchCount;
            size_t const off = (size_t)(b - gpuMF->chunkStartBlock) * ZSTD_GPU_BLOCKSIZE;
            size_t const sz = (gpuMF->chunkSize - off < ZSTD_GPU_BLOCKSIZE)
                            ? (gpuMF->chunkSize - off)
                            : (size_t)ZSTD_GPU_BLOCKSIZE;
            int const inBatch = ZSTD_gpuBatch_addBlock(gpuMF, (const char*)gpuMF->chunkSrc + off, sz);
            if (inBatch < 0) return ERROR(GPU_allocationFailed);
        }
    }
    return 0;
}

/* Stage the look-ahead and wait for the next block's sequences (in order).
 * The current block is gpuMF->nextWait (advanced per call). */
size_t ZSTD_gpuChunk_processBlock(ZSTD_GpuMatchFinder* gpuMF, ZSTD_GpuSequence* dst,
                                  size_t dstCapacity)
{
    if (!gpuMF || !dst) return ERROR(GPU_invalidParameter);
    {
        unsigned const g = gpuMF->nextWait;
        FORWARD_IF_ERROR(gpuChunk_stageLookahead(gpuMF, g), "GPU chunk stage failed");
        /* If the current block is still in the unsubmitted batch (the look-ahead
         * was limited by the chunk end), submit it (partial) so the GPU has it. */
        if (g >= gpuMF->curBatchStart) {
            FORWARD_IF_ERROR(gpuChunk_submitBatch(gpuMF), "GPU chunk batch submit failed");
        }
        /* Map the global block index to (batchIdx, in-batch index) via the
         * 2-slot submitted-batch history. */
        {
            unsigned batchIdx = 0, blockInBatch = 0, slot = 0;
            int found = 0;
            for (int i = 0; i < 2; i++) {
                if (g >= gpuMF->histStart[i] && g < gpuMF->histStart[i] + gpuMF->histCount[i]) {
                    batchIdx = gpuMF->histBatchIdx[i];
                    blockInBatch = g - gpuMF->histStart[i];
                    slot = (unsigned)i;
                    found = 1;
                    break;
                }
            }
            if (!found) return ERROR(GPU_invalidParameter);
            /* Waiting on the batch's last block frees a pipeline slot. */
            if (blockInBatch + 1 == gpuMF->histCount[slot]) gpuMF->inFlight--;
            gpuMF->nextWait++;
            return ZSTD_gpuBatch_waitBlock(gpuMF, batchIdx, blockInBatch, dst, dstCapacity, NULL, 0);
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
        size_t const err = ZSTD_gpuBatch_submit(gpuMF);
        if (ZSTD_isError(err)) return err;
        /* The block is inBatch of the just-submitted batch (batchIdx-1). */
        unsigned const batchIdx = gpuMF->batchIdx - 1;
        return ZSTD_gpuBatch_waitBlock(gpuMF, batchIdx, (unsigned)inBatch,
                                       (ZSTD_GpuSequence*)dst, dstCapacity, src, srcSize);
    }
}
