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
 *      compute     : wait -> fillHist (fingerprint + per-hash histogram) ->
 *                    row scan (exclusive prefix -> row starts) -> scatter
 *                    (positions into per-hash rows, in position order) ->
 *                    search (candidate = max(in-batch row
 *                    predecessor, pre-batch table entry)) -> insert
 *                    (atomicMax table update for the next batch). The row
 *                    construction is a single counting sort - no radix (G6).
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
/* The greedy extends each match with the whole CUDA block (one block per
 * frame block) : ZSTD_GPU_GREEDY_THREADS x 8 B per round, so a whole-block
 * match is ~64 rounds instead of the old single-warp ~512. The rounds are
 * pipelined (the next round's loads are issued while the current round's
 * ballots are aggregated), breaking the dependent load-latency chain. */
#define ZSTD_GPU_GREEDY_THREADS 256u
#define ZSTD_GPU_GREEDY_WARPS   (ZSTD_GPU_GREEDY_THREADS / 32u)
#define ZSTD_GPU_GREEDY_ROUND   (ZSTD_GPU_GREEDY_THREADS * 8u)
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

/* Sort-free device search (G6): one thread per valid position. Computes the
 * fingerprint once, packs (absolute position, hash) into d_posHash, and builds
 * the per-hash histogram in the same pass. Replaces the old fillKeys (which
 * wrote composite sort keys for the radix) and the radix's histogram pass.
 * The scatter copies d_posHash into the row array, so the search reads the
 * predecessor's (absPos, hash) from one 8-byte load, and the search/insert
 * neither re-hash the window nor re-run the block-locate loop. */
__global__ void gpuFillHist(const unsigned char* __restrict__ d_window, unsigned winBase,
                            const unsigned* __restrict__ d_blockMeta, unsigned B,
                            unsigned hashLog, unsigned mls,
                            unsigned long long* __restrict__ d_posHash,
                            unsigned* __restrict__ d_hist,
                            unsigned totalPos)
{
    unsigned const t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= totalPos) return;
    unsigned i, a, pib;
    gpuBatchLocate(d_blockMeta, B, t, i, a, pib);
    unsigned char const* const p = d_window + (a - winBase);
    unsigned const h = gpuHash(p, hashLog, mls);
    d_posHash[t] = ((unsigned long long)a << 32) | (unsigned long long)h;
    atomicAdd(&d_hist[h], 1);
}

/* ======================================================================== */
/* Sort-free row construction (G6)                                          */
/*                                                                          */
/* The radix sort (2 passes x {hist, scanL1, scanL2, scatter}) is replaced  */
/* by a single counting sort over the whole fingerprint:                    */
/*   1. gpuFillHist (above) builds the per-hash count d_hist.               */
/*   2. the row scan turns d_hist into the exclusive row starts             */
/*      (d_rowStart) with a two-level block scan.                           */
/*   3. gpuScatter places each position into its row. The slot is taken     */
/*      with atomicAdd, so the row order is the order in which the          */
/*      same-hash atomicAdds complete - which follows thread (i.e.          */
/*      position) order: warps run in lane order, blocks in grid order.     */
/*      The row is therefore in position order up to rare cross-warp/       */
/*      block reordering, and the in-batch predecessor of a position is     */
/*      the previous element of the row (the search checks the order and    */
/*      falls back to the pre-batch table entry on the rare inversion).     */
/* This removes the radix's 8 kernel launches and the recordSlot pass, and  */
/* makes the search/insert hash-free (they reuse d_hash/d_absPos).          */
/* ======================================================================== */

/* Hillis-Steele inclusive block scan of 1024 lanes (shared memory). The
 * caller loads s_scan[threadIdx.x] and synchronizes before calling; after
 * the call s_scan[i] holds the inclusive prefix sum of lanes [0..i]. */
__device__ __forceinline__ void gpuBlockScan1024(unsigned* s_scan)
{
    for (unsigned stride = 1; stride < 1024; stride <<= 1) {
        __syncthreads();
        unsigned const up = (threadIdx.x >= stride) ? s_scan[threadIdx.x - stride] : 0;
        s_scan[threadIdx.x] += up;
    }
    __syncthreads();
}

/* Level 1 of the row scan: one CUDA block of 1024 threads over 1024
 * consecutive histogram entries. d_rowStart receives the within-block
 * exclusive prefix; d_blockTotal receives the block sum. */
__global__ void gpuRowScanBlock(const unsigned* __restrict__ d_hist, unsigned n,
                                unsigned* __restrict__ d_rowStart,
                                unsigned* __restrict__ d_blockTotal)
{
    __shared__ unsigned s_scan[1024];
    unsigned const e = blockIdx.x * 1024u + threadIdx.x;
    s_scan[threadIdx.x] = (e < n) ? d_hist[e] : 0;
    gpuBlockScan1024(s_scan);
    if (e < n) d_rowStart[e] = (threadIdx.x == 0) ? 0 : s_scan[threadIdx.x - 1];
    if (threadIdx.x == 1023) d_blockTotal[blockIdx.x] = s_scan[1023];
}

/* Level 2a: nScanBlocks <= 1024 - one block scans the block totals. */
__global__ void gpuRowScanGlobalSmall(const unsigned* __restrict__ d_blockTotal,
                                      unsigned nBlocks,
                                      unsigned* __restrict__ d_blockPrefix)
{
    __shared__ unsigned s_scan[1024];
    s_scan[threadIdx.x] = (threadIdx.x < nBlocks) ? d_blockTotal[threadIdx.x] : 0;
    gpuBlockScan1024(s_scan);
    if (threadIdx.x < nBlocks)
        d_blockPrefix[threadIdx.x] = (threadIdx.x == 0) ? 0 : s_scan[threadIdx.x - 1];
}

/* Level 2b-i: nScanBlocks > 1024 - per-sub-block scan of the block totals
 * (1024 per CUDA block). d_blockPrefix gets the within-sub-block exclusive
 * prefix; d_subTotal gets the sub-block sum. */
__global__ void gpuRowScanSubBlock(const unsigned* __restrict__ d_blockTotal,
                                   unsigned nBlocks,
                                   unsigned* __restrict__ d_blockPrefix,
                                   unsigned* __restrict__ d_subTotal)
{
    __shared__ unsigned s_scan[1024];
    unsigned const e = blockIdx.x * 1024u + threadIdx.x;
    s_scan[threadIdx.x] = (e < nBlocks) ? d_blockTotal[e] : 0;
    gpuBlockScan1024(s_scan);
    if (e < nBlocks)
        d_blockPrefix[e] = (threadIdx.x == 0) ? 0 : s_scan[threadIdx.x - 1];
    if (threadIdx.x == 1023) d_subTotal[blockIdx.x] = s_scan[1023];
}

/* Level 2b-ii: one block scans the sub-block totals (<= 1024 of them). */
__global__ void gpuRowScanSubGlobal(const unsigned* __restrict__ d_subTotal,
                                    unsigned nSubBlocks,
                                    unsigned* __restrict__ d_subPrefix)
{
    __shared__ unsigned s_scan[1024];
    s_scan[threadIdx.x] = (threadIdx.x < nSubBlocks) ? d_subTotal[threadIdx.x] : 0;
    gpuBlockScan1024(s_scan);
    if (threadIdx.x < nSubBlocks)
        d_subPrefix[threadIdx.x] = (threadIdx.x == 0) ? 0 : s_scan[threadIdx.x - 1];
}

/* Level 2b-iii: fold the sub-block prefix into the per-block prefix. */
__global__ void gpuRowScanCombine(unsigned nBlocks,
                                  const unsigned* __restrict__ d_subPrefix,
                                  unsigned* __restrict__ d_blockPrefix)
{
    unsigned const e = blockIdx.x * 1024u + threadIdx.x;
    if (e < nBlocks) d_blockPrefix[e] += d_subPrefix[e >> 10];
}

/* Finish the row scan: add the global (per-block) prefix to the within-block
 * exclusive prefixes, leaving d_rowStart as the full exclusive prefix of
 * d_hist - i.e. the start index of each hash row in the sorted array. */
__global__ void gpuRowScanAdd(unsigned n,
                              const unsigned* __restrict__ d_blockPrefix,
                              unsigned* __restrict__ d_rowStart)
{
    unsigned const e = blockIdx.x * 1024u + threadIdx.x;
    if (e < n) d_rowStart[e] += d_blockPrefix[e >> 10];
}

/* Scatter: place each position's (absPos, hash) word into its hash row
 * (d_sorted, the d_keyA buffer). The search then walks the array in index
 * order and reads each element's row predecessor as the previous array
 * element, so no per-position slot mapping is needed. d_rowCounter must be
 * zeroed before the launch. The atomicAdd order follows thread order (see the
 * section comment), so the rows come out in position order up to rare
 * reordering. */
__global__ void gpuScatter(unsigned totalPos,
                           const unsigned long long* __restrict__ d_posHash,
                           const unsigned* __restrict__ d_rowStart,
                           unsigned* __restrict__ d_rowCounter,
                           unsigned long long* __restrict__ d_sorted,
                           unsigned* __restrict__ d_slot)
{
    unsigned const t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= totalPos) return;
    unsigned const h = (unsigned)d_posHash[t];
    unsigned const slot = atomicAdd(&d_rowCounter[h], 1);
    unsigned const dst = d_rowStart[h] + slot;
    d_sorted[dst] = d_posHash[t];
    d_slot[t] = dst;
}

/* The search (sort-free, G6): for each valid position, the in-batch
 * predecessor is the previous element of its hash row (d_sorted[j-1], where
 * j = d_slot[t]). The row is in position order up to rare cross-warp/block
 * reordering (see the row-construction comment), so that element is the
 * closest previous same-hash position in the batch - the same candidate the
 * CPU's hash table yields. Each row element packs (absPos, hash), so the
 * search verifies the same-hash and earlier-position conditions from a single
 * 8-byte load. On the rare inversion (the previous element is a different
 * hash or a later position) the search falls back to the pre-batch table
 * entry, which is still a valid same-hash candidate. The candidate is the max
 * of the two, and must pass the window-age check and the 8-byte prefilter
 * before the (capped) count. The frame-block index and position-within-block
 * are derived from the absolute position (the batch starts at batchStart and
 * the blocks are BLOCKSIZE apart). The result is written directly to
 * d_cand[block][posInBlock] = (offset, len); offset == 0 means no candidate.
 * The hash and absolute position come from the gpuFillHist cache (d_posHash),
 * so the search neither re-hashes the window nor re-runs the block-locate
 * loop. */
__global__ void gpuSearch(const unsigned char* __restrict__ d_window, unsigned winBase,
                          const unsigned long long* __restrict__ d_posHash,
                          const unsigned long long* __restrict__ d_sorted,
                          const unsigned* __restrict__ d_slot,
                          const unsigned* __restrict__ d_table,
                          unsigned B, unsigned batchStart, unsigned windowSize,
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
    /* The thread index is the in-batch position t, so the d_posHash/d_slot
     * reads and the d_cand writes are sequential (coalesced). The row
     * predecessor is a random d_sorted read (j = d_slot[t]); that is the
     * rnd-data cost, but the coalesced d_cand writes matter more on periodic
     * data, where every position is a candidate. */
    unsigned const t = blockIdx.x * blockDim.x + threadIdx.x;
    int isCand = 0;
    unsigned i = 0;
    if (t < totalPos) {
        unsigned long long const packed = d_posHash[t];
        unsigned const a = (unsigned)(packed >> 32);
        unsigned const h = (unsigned)packed;

        /* In-batch predecessor: the previous element of the row (j-1), if it
         * is the same hash and a genuinely earlier position. The row is in
         * position order up to rare cross-warp/block reordering, in which case
         * the previous element is a later position and is rejected by the
         * absPos test. Position 0 is a valid absolute position, so track
         * validity separately. */
        unsigned q1 = 0; int hasQ1 = 0;
        {
            unsigned const j = d_slot[t];
            if (j > 0) {
                unsigned long long const prev = d_sorted[j - 1];
                unsigned const prevAbs = (unsigned)(prev >> 32);
                if ((unsigned)prev == h && prevAbs < a) {
                    q1 = prevAbs;
                    hasQ1 = 1;
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
            /* 8-byte prefilter (G12.1): two 64-bit loads at the candidate
             * head. Valid positions always have 8 bytes left in the block
             * (nValid = size-7), so the read stays inside the block. For the
             * GPU levels minMatch <= 8, so a passing prefilter means the
             * candidate is at least minMatch long; the capped length is
             * ZSTD_GPU_MATCH_CAP and the greedy extends it. */
            pass = (gpuRead64(d_window + (a - winBase)) == gpuRead64(qp));
        }
        if (pass) {
            /* Frame-block index and position-within-block from the absolute
             * position (the batch starts at batchStart, blocks are BLOCKSIZE
             * apart, a power of two). */
            unsigned const off = a - batchStart;
            i = off >> 17;
            unsigned const pib = off & (ZSTD_GPU_BLOCKSIZE - 1);
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
 * position per fingerprint). Uses the gpuFillHist cache (d_posHash) - no
 * re-hashing, no block-locate. Runs after the search so the search saw the
 * pre-batch table. */
__global__ void gpuInsert(unsigned totalPos,
                          const unsigned long long* __restrict__ d_posHash,
                          unsigned* __restrict__ d_table)
{
    unsigned const t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= totalPos) return;
    unsigned long long const packed = d_posHash[t];
    unsigned const a = (unsigned)(packed >> 32);
    unsigned const h = (unsigned)packed;
    atomicMax(d_table + h, a + 1);   /* store pos+1; a+1 is the latest so far */
}

/* ======================================================================== */
/* Device greedy (G18.1, cheapened for long matches) : one CUDA block of
 * ZSTD_GPU_GREEDY_THREADS (256) threads per frame block walks the positions in
 * order and emits non-overlapping matches, jumping past each emitted match
 * (pib += len) instead of stepping through the covered positions. Thread 0
 * drives the sequential walk (finding the match-starts); the whole block
 * cooperatively extends each match : 256 threads x 8 bytes = 2 KB per round,
 * so a whole-block match is ~64 rounds of coalesced 64-bit loads (the old
 * single-warp walk was 32 lanes x 8 B = 256 B/round, ~512 rounds, and each
 * round waited on the previous round's loads - a dependent latency chain that
 * dominated the periodic-data cost). The rounds are now pipelined : the next
 * round's loads are issued before the current round's ballots are aggregated,
 * so two rounds are in flight and the chain is broken.
 *
 * The first mismatch is a block-wide concept (the smallest byte position that
 * differs) but __ballot_sync is per-warp : each warp records its ballot in
 * shared memory (s_warpBallot) and the block scans the 8 ballots for the first
 * non-zero warp, then the first lane within it. That thread byte-scans its
 * 8-byte chunk to the exact position.
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
    __shared__ unsigned s_pib, s_off, s_state;   /* state : 0 = walk, 1 = done */
    __shared__ unsigned s_warpBallot[ZSTD_GPU_GREEDY_WARPS];
    __shared__ unsigned s_exactPos;
    unsigned const size = d_blockMeta[i*2+1];
    unsigned const absStart = d_blockMeta[i*2];
    if (threadIdx.x == 0) {
        s_pib = 0; s_off = 0;
        s_state = (d_nbCand[i] == 0) ? 1 : 0;
    }
    __syncthreads();
    if (s_state == 1) { if (threadIdx.x == 0) d_nbSeq[i] = 0; return; }

    unsigned const t = threadIdx.x;
    unsigned const warpId = t >> 5;
    unsigned const lane = t & 31u;
    unsigned nbSeq = 0;
    for (;;) {
        /* Phase A : thread 0 finds the next match-start (or marks done). */
        if (t == 0) {
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
        /* Phase B : the block cooperatively extends the match at pib.
         * The first 8 bytes are known equal (the search's prefilter), so the
         * extension covers [8, maxLen). Round r covers [8+GR*r, 8+GR*(r+1));
         * thread t checks the 8 bytes at [8+GR*r+8t, +8). */
        unsigned const a = absStart + pib;
        unsigned const maxLen = size - pib;
        const unsigned char* const base = d_window + (a - winBase);
        const unsigned char* const srcp = d_window + (a - off - winBase);
        unsigned len = ZSTD_GPU_MATCH_CAP;   /* 8 : the prefilter already matched */
        /* Full rounds : [len, len+GR) fully in range. Pipelined : the next
         * round's loads (bn, sn) are issued before the current round's ballots
         * are aggregated, so the current round's loads were issued last
         * iteration and are in flight while this round is checked. */
        unsigned done = 0;
        unsigned long long b = 0, s = 0;
        if (len + ZSTD_GPU_GREEDY_ROUND <= maxLen) {
            unsigned const off0 = len + t*8;
            b = gpuRead64(base + off0);
            s = gpuRead64(srcp + off0);
        }
        while (!done && len + ZSTD_GPU_GREEDY_ROUND <= maxLen) {
            unsigned long long bn = 0, sn = 0;
            if (len + 2*ZSTD_GPU_GREEDY_ROUND <= maxLen) {
                unsigned const offN = len + ZSTD_GPU_GREEDY_ROUND + t*8;
                bn = gpuRead64(base + offN);
                sn = gpuRead64(srcp + offN);
            }
            unsigned const bal = __ballot_sync(0xFFFFFFFFu, b != s);
            if (lane == 0) s_warpBallot[warpId] = bal;
            __syncthreads();
            unsigned firstWarp = ZSTD_GPU_GREEDY_WARPS;
            #pragma unroll
            for (unsigned w = 0; w < ZSTD_GPU_GREEDY_WARPS; w++) {
                if (firstWarp == ZSTD_GPU_GREEDY_WARPS && s_warpBallot[w] != 0) firstWarp = w;
            }
            if (firstWarp == ZSTD_GPU_GREEDY_WARPS) {
                len += ZSTD_GPU_GREEDY_ROUND;   /* all 2 KB equal : advance */
                b = bn; s = sn;                 /* the prefetch becomes current */
            } else {
                unsigned const fl = (unsigned)__ffs(s_warpBallot[firstWarp]) - 1;
                unsigned const ft = firstWarp * 32 + fl;
                if (t == ft) {
                    unsigned local = len + ft*8;
                    while (local < maxLen && base[local] == srcp[local]) local++;
                    s_exactPos = local;
                }
                __syncthreads();
                len = s_exactPos;
                done = 1;
            }
        }
        /* Tail : [len, maxLen) is < GR bytes (only if no mismatch in the full
         * rounds). 256 threads x 8 B covers it in one round ; the last thread
         * may be partially or fully out of range (its bytes are "equal"). */
        if (!done) {
            unsigned const offT = len + t*8;
            unsigned mismatch = 0;
            if (offT < maxLen) {
                if (offT + 8 <= maxLen) {
                    unsigned long long const bb = gpuRead64(base + offT);
                    unsigned long long const ss = gpuRead64(srcp + offT);
                    mismatch = (bb != ss);
                } else {
                    unsigned local = offT;
                    while (local < maxLen && base[local] == srcp[local]) local++;
                    mismatch = (local < maxLen);
                }
            }
            unsigned const bal = __ballot_sync(0xFFFFFFFFu, mismatch);
            if (lane == 0) s_warpBallot[warpId] = bal;
            __syncthreads();
            unsigned firstWarp = ZSTD_GPU_GREEDY_WARPS;
            #pragma unroll
            for (unsigned w = 0; w < ZSTD_GPU_GREEDY_WARPS; w++) {
                if (firstWarp == ZSTD_GPU_GREEDY_WARPS && s_warpBallot[w] != 0) firstWarp = w;
            }
            if (firstWarp == ZSTD_GPU_GREEDY_WARPS) {
                if (t == 0) s_exactPos = maxLen;   /* the whole tail is equal */
            } else {
                unsigned const fl = (unsigned)__ffs(s_warpBallot[firstWarp]) - 1;
                unsigned const ft = firstWarp * 32 + fl;
                if (t == ft) {
                    unsigned local = len + ft*8;
                    while (local < maxLen && base[local] == srcp[local]) local++;
                    s_exactPos = local;
                }
            }
            __syncthreads();
            len = s_exactPos;
        }
        /* Phase C : thread 0 emits the sequence and advances the walk. */
        if (t == 0) {
            ZSTD_GpuSequence* const sq = d_seq + (i * ZSTD_GPU_MAX_SEQS + nbSeq);
            sq->position = pib;
            sq->offset = off;
            sq->matchLength = len;
            nbSeq++;
            s_pib = pib + len;   /* jump past the emitted match */
        }
        __syncthreads();
    }
    if (t == 0) d_nbSeq[i] = nbSeq;
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
        bytes += (size_t)B * ZSTD_GPU_BLOCKSIZE * 8;                       /* keyA (row array) */
        bytes += (size_t)B * ZSTD_GPU_BLOCKSIZE * 4;                       /* slot */
        bytes += (size_t)B * ZSTD_GPU_BLOCKSIZE * 8;                       /* posHash */
        bytes += (size_t)(1u << hashLog) * 4 * 3;                          /* hist/rowStart/rowCounter */
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
            /* Sort-free buffers (G6). d_keyA doubles as the 32-bit row array
             * (d_sorted); d_keyB and the radix scratch buffers are no longer
             * allocated (they stay NULL and the free path is NULL-checked). */
            size_t const npos = (size_t)mf->B * ZSTD_GPU_BLOCKSIZE;   /* max valid positions per batch */
            size_t const hsize = (size_t)(1u << mf->hashLog) * 4;     /* per-hash buffer */
            if (cudaMalloc(&mf->d_keyA, npos * 8) != cudaSuccess) goto fail;
            if (cudaMalloc(&mf->d_slot, npos * 4) != cudaSuccess) goto fail;
            if (cudaMalloc(&mf->d_posHash, npos * 8) != cudaSuccess) goto fail;
            if (cudaMalloc(&mf->d_hist, hsize) != cudaSuccess) goto fail;
            if (cudaMalloc(&mf->d_rowStart, hsize) != cudaSuccess) goto fail;
            if (cudaMalloc(&mf->d_rowCounter, hsize) != cudaSuccess) goto fail;
            {
                unsigned const nScanBlocks = (mf->hashLog > 10) ? (1u << (mf->hashLog - 10)) : 1u;
                unsigned const nSubBlocks = (nScanBlocks > 1024) ? (nScanBlocks + 1023) / 1024 : 1u;
                if (cudaMalloc(&mf->d_blockTotal, (size_t)nScanBlocks * 4) != cudaSuccess) goto fail;
                if (cudaMalloc(&mf->d_blockPrefix, (size_t)nScanBlocks * 4) != cudaSuccess) goto fail;
                if (cudaMalloc(&mf->d_subTotal, (size_t)nSubBlocks * 4) != cudaSuccess) goto fail;
                if (cudaMalloc(&mf->d_subPrefix, (size_t)nSubBlocks * 4) != cudaSuccess) goto fail;
            }
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
                /* G20.0 : double-buffered by batch parity (2 x 16 events). */
                for (int b = 0; b < 2; b++) for (int p = 0; p < 8; p++) {
                    if (cudaEventCreate(&mf->profEv[b][2*p]) != cudaSuccess) goto fail;
                    if (cudaEventCreate(&mf->profEv[b][2*p+1]) != cudaSuccess) goto fail;
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
        "H2D", "fillHist", "rowScan", "scatter",
        "search", "insert", "greedy", "D2H"};
    double total = 0;
    for (int p = 0; p < 8; p++) total += mf->phaseMs[p];
    fprintf(stderr, "[ZSTD_GPU_PROFILE] per batch of %u blocks (avg over %u batches)\n",
            mf->B, (unsigned)mf->phaseCount);
    for (int p = 0; p < 8; p++) {
        double avg = mf->phaseMs[p] / (double)mf->phaseCount;
        double pct = (total > 0) ? 100.0 * avg / (total / (double)mf->phaseCount) : 0.0;
        fprintf(stderr, "  %-11s %9.4f ms  %5.1f%%\n", names[p], avg, pct);
    }
    fprintf(stderr, "  %-11s %9.4f ms\n", "total", total / (double)mf->phaseCount);
    fprintf(stderr, "[ZSTD_GPU_PROFILE] per 128KB block: total %.4f ms\n",
            total / (double)mf->phaseCount / (double)mf->B);
}

size_t ZSTD_freeGpuMatchFinder(ZSTD_GpuMatchFinder* gpuMF)
{
    if (!gpuMF) return 0;
    gpuProfReport(gpuMF);
    ZSTD_gpuHostProfReport();   /* G20 : host-side phases, if enabled */
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
    if (gpuMF->d_posHash) cudaFree(gpuMF->d_posHash);
    if (gpuMF->d_hist) cudaFree(gpuMF->d_hist);
    if (gpuMF->d_rowStart) cudaFree(gpuMF->d_rowStart);
    if (gpuMF->d_rowCounter) cudaFree(gpuMF->d_rowCounter);
    if (gpuMF->d_blockTotal) cudaFree(gpuMF->d_blockTotal);
    if (gpuMF->d_blockPrefix) cudaFree(gpuMF->d_blockPrefix);
    if (gpuMF->d_subTotal) cudaFree(gpuMF->d_subTotal);
    if (gpuMF->d_subPrefix) cudaFree(gpuMF->d_subPrefix);
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
    if (gpuMF->profile) for (int b = 0; b < 2; b++) for (int p = 0; p < 8; p++) {
        if (gpuMF->profEv[b][2*p]) cudaEventDestroy(gpuMF->profEv[b][2*p]);
        if (gpuMF->profEv[b][2*p+1]) cudaEventDestroy(gpuMF->profEv[b][2*p+1]);
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
 *   compute      : wait -> fillHist -> rowScan -> scatter -> search -> insert
 *   copy stream  : wait -> D2H of the per-position candidates
 * The batch state is reset so the next batch can be staged. */
/* G17 per-phase profiling helpers (no-ops unless ZSTD_GPU_PROFILE).
 * G20.0 : the events are double-buffered by batch parity (profEv[buf]) : with
 * the chunk pipeline the next batch is submitted (and records new events)
 * before the previous batch's last block is waited on, so a single event set
 * would be overwritten before gpuProfAccum reads it. At most 2 batches are in
 * flight, so parity is safe : batch k's events are read at the wait of batch
 * k's last block, and batch k+2 is only submitted afterwards. */
static void gpuProfRec(ZSTD_GpuMatchFinder* mf, int buf, int idx, cudaStream_t st)
{
    if (mf->profile) (void)cudaEventRecord(mf->profEv[buf][idx], st);
}

static void gpuProfAccum(ZSTD_GpuMatchFinder* mf, int buf)
{
    if (!mf->profile) return;
    for (int p = 0; p < 8; p++) {
        float ms = 0.f;
        (void)cudaEventElapsedTime(&ms, mf->profEv[buf][2*p], mf->profEv[buf][2*p+1]);
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
        gpuProfRec(gpuMF, buf, 0, gpuMF->copyStream);   /* H2D phase start */
        if (cudaMemcpyAsync((unsigned char*)gpuMF->d_window + gpuMF->winUsed,
                            gpuMF->h_staging[buf], totalBytes,
                            cudaMemcpyHostToDevice, gpuMF->copyStream) != cudaSuccess)
            return ERROR(GPU_cudaMemcpyFailed);
        gpuMF->winUsed += totalBytes;
        if (cudaEventRecord(gpuMF->evH2d[buf], gpuMF->copyStream) != cudaSuccess)
            return ERROR(GPU_cudaMemcpyFailed);
        gpuProfRec(gpuMF, buf, 1, gpuMF->copyStream);   /* H2D phase end */

        /* Compute: wait for the H2D, then run the pipeline. */
        if (cudaStreamWaitEvent(gpuMF->stream, gpuMF->evH2d[buf], 0) != cudaSuccess)
            return ERROR(GPU_cudaMemcpyFailed);
        {
            unsigned totalPos = 0;
            for (unsigned i = 0; i < count; i++) {
                unsigned const sz = gpuMF->batchSize[i];
                totalPos += (sz >= 8) ? (sz - 7) : 0;
            }
            unsigned const grid = (totalPos + ZSTD_GPU_THREADS - 1) / ZSTD_GPU_THREADS;
            unsigned const nHist = 1u << gpuMF->hashLog;
            unsigned const nScanBlocks = (nHist + 1023u) / 1024u;
            unsigned const nSubBlocks = (nScanBlocks + 1023u) / 1024u;

            /* Sort-free pipeline (G6): fill+hist -> row scan -> scatter ->
             * search -> insert -> greedy. Replaces fillKeys -> 2-pass radix ->
             * recordSlot. The row scan is a two-level exclusive prefix of the
             * per-hash histogram (d_hist -> d_rowStart). */
            gpuProfRec(gpuMF, buf, 2, gpuMF->stream);
            if (cudaMemsetAsync(gpuMF->d_hist, 0, (size_t)nHist * 4, gpuMF->stream) != cudaSuccess)
                return ERROR(GPU_cudaMemcpyFailed);
            gpuFillHist<<<grid, ZSTD_GPU_THREADS, 0, gpuMF->stream>>>(
                (unsigned char*)gpuMF->d_window, gpuMF->winBase,
                (unsigned*)gpuMF->d_blockMeta, count,
                gpuMF->hashLog, gpuMF->mls,
                (unsigned long long*)gpuMF->d_posHash,
                (unsigned*)gpuMF->d_hist, totalPos);
            gpuProfRec(gpuMF, buf, 3, gpuMF->stream);

            gpuProfRec(gpuMF, buf, 4, gpuMF->stream);
            gpuRowScanBlock<<<nScanBlocks, 1024, 0, gpuMF->stream>>>(
                (unsigned*)gpuMF->d_hist, nHist,
                (unsigned*)gpuMF->d_rowStart, (unsigned*)gpuMF->d_blockTotal);
            if (nScanBlocks <= 1024) {
                gpuRowScanGlobalSmall<<<1, 1024, 0, gpuMF->stream>>>(
                    (unsigned*)gpuMF->d_blockTotal, nScanBlocks,
                    (unsigned*)gpuMF->d_blockPrefix);
            } else {
                gpuRowScanSubBlock<<<nSubBlocks, 1024, 0, gpuMF->stream>>>(
                    (unsigned*)gpuMF->d_blockTotal, nScanBlocks,
                    (unsigned*)gpuMF->d_blockPrefix, (unsigned*)gpuMF->d_subTotal);
                gpuRowScanSubGlobal<<<1, 1024, 0, gpuMF->stream>>>(
                    (unsigned*)gpuMF->d_subTotal, nSubBlocks,
                    (unsigned*)gpuMF->d_subPrefix);
                gpuRowScanCombine<<<nSubBlocks, 1024, 0, gpuMF->stream>>>(
                    nScanBlocks, (unsigned*)gpuMF->d_subPrefix,
                    (unsigned*)gpuMF->d_blockPrefix);
            }
            gpuRowScanAdd<<<nScanBlocks, 1024, 0, gpuMF->stream>>>(
                nHist, (unsigned*)gpuMF->d_blockPrefix,
                (unsigned*)gpuMF->d_rowStart);
            gpuProfRec(gpuMF, buf, 5, gpuMF->stream);

            /* Scatter the positions into their hash rows (records d_slot). */
            gpuProfRec(gpuMF, buf, 6, gpuMF->stream);
            if (cudaMemsetAsync(gpuMF->d_rowCounter, 0, (size_t)nHist * 4, gpuMF->stream) != cudaSuccess)
                return ERROR(GPU_cudaMemcpyFailed);
            gpuScatter<<<grid, ZSTD_GPU_THREADS, 0, gpuMF->stream>>>(
                totalPos,
                (unsigned long long*)gpuMF->d_posHash, (unsigned*)gpuMF->d_rowStart,
                (unsigned*)gpuMF->d_rowCounter,
                (unsigned long long*)gpuMF->d_keyA, (unsigned*)gpuMF->d_slot);
            gpuProfRec(gpuMF, buf, 7, gpuMF->stream);

            /* Clear the batch's candidate slots before the search. The search
             * only writes positions that yield a valid candidate, and the
             * greedy treats any non-zero offset as a match : stale data from a
             * previous batch must not leak through as a phantom match. */
            if (cudaMemsetAsync(gpuMF->d_cand, 0, (size_t)count * ZSTD_GPU_BLOCKSIZE * 8, gpuMF->stream) != cudaSuccess)
                return ERROR(GPU_cudaMemcpyFailed);
            if (cudaMemsetAsync(gpuMF->d_nbCand, 0, (size_t)count * 4, gpuMF->stream) != cudaSuccess)
                return ERROR(GPU_cudaMemcpyFailed);

            gpuProfRec(gpuMF, buf, 8, gpuMF->stream);
            gpuSearch<<<grid, ZSTD_GPU_THREADS, 0, gpuMF->stream>>>(
                (unsigned char*)gpuMF->d_window, gpuMF->winBase,
                (unsigned long long*)gpuMF->d_posHash,
                (unsigned long long*)gpuMF->d_keyA,
                (unsigned*)gpuMF->d_slot,
                (unsigned*)gpuMF->d_table,
                count, gpuMF->framePos, gpuMF->windowSize, totalPos,
                (unsigned*)gpuMF->d_cand, (unsigned*)gpuMF->d_nbCand);
            gpuProfRec(gpuMF, buf, 9, gpuMF->stream);   /* search end */

            gpuProfRec(gpuMF, buf, 10, gpuMF->stream);   /* insert start */
            gpuInsert<<<grid, ZSTD_GPU_THREADS, 0, gpuMF->stream>>>(
                totalPos,
                (unsigned long long*)gpuMF->d_posHash,
                (unsigned*)gpuMF->d_table);
            gpuProfRec(gpuMF, buf, 11, gpuMF->stream);   /* insert end */

            gpuProfRec(gpuMF, buf, 12, gpuMF->stream);
            gpuGreedy<<<count, ZSTD_GPU_GREEDY_THREADS, 0, gpuMF->stream>>>(
                (unsigned char*)gpuMF->d_window, gpuMF->winBase,
                (unsigned*)gpuMF->d_blockMeta, count,
                (unsigned*)gpuMF->d_cand,
                (unsigned*)gpuMF->d_nbCand,
                (ZSTD_GpuSequence*)gpuMF->d_seq[buf],
                (unsigned*)gpuMF->d_nbSeq[buf]);
            gpuProfRec(gpuMF, buf, 13, gpuMF->stream);
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
        gpuProfRec(gpuMF, buf, 14, gpuMF->copyStream);
        if (cudaMemcpyAsync(gpuMF->h_nbSeq[buf], gpuMF->d_nbSeq[buf],
                            (size_t)count * 4,
                            cudaMemcpyDeviceToHost, gpuMF->copyStream) != cudaSuccess)
            return ERROR(GPU_cudaMemcpyFailed);
        if (cudaEventRecord(gpuMF->evNbSeq[buf], gpuMF->copyStream) != cudaSuccess)
            return ERROR(GPU_cudaMemcpyFailed);
        gpuProfRec(gpuMF, buf, 15, gpuMF->copyStream);

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
        {
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
        }

        /* G17: accumulate the per-phase times now that all of this batch's
         * GPU work is complete (the events were recorded during submit). Done
         * once per batch, on its last block. The events are double-buffered by
         * batch parity (G20.0), so the next batch's submit cannot overwrite
         * them before this read. */
        if (blockInBatch + 1 >= info->count) gpuProfAccum(gpuMF, (int)slot);

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
        double const p0 = ZSTD_gpuHostNow();
        FORWARD_IF_ERROR(gpuChunk_stageLookahead(gpuMF, g), "GPU chunk stage failed");
        double const p1 = ZSTD_gpuHostNow();
        /* If the current block is still in the unsubmitted batch (the look-ahead
         * was limited by the chunk end), submit it (partial) so the GPU has it. */
        if (g >= gpuMF->curBatchStart) {
            FORWARD_IF_ERROR(gpuChunk_submitBatch(gpuMF), "GPU chunk batch submit failed");
        }
        double const p2 = ZSTD_gpuHostNow();
        ZSTD_gpuHostProfPhase(3, p0, p1);   /* G20.2 : staging */
        ZSTD_gpuHostProfPhase(4, p1, p2);   /* G20.2 : submit */
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
            double const w0 = ZSTD_gpuHostNow();
            size_t const r = ZSTD_gpuBatch_waitBlock(gpuMF, batchIdx, blockInBatch, dst, dstCapacity, NULL, 0);
            ZSTD_gpuHostProfPhase(5, w0, ZSTD_gpuHostNow());   /* G20.2 : wait */
            return r;
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
