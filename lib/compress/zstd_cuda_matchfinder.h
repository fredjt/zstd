/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */

#ifndef ZSTD_CUDA_MATCHFINDER_H
#define ZSTD_CUDA_MATCHFINDER_H

#include "zstd.h"                             /* ZSTD_Sequence */
#include "zstd_compress_internal.h"           /* SeqDef, ZSTD_CCtx_params */
#include "../common/zstd_deps.h"              /* size_t, NULL */
#include "../common/zstd_internal.h"          /* ZSTD_compressionParameters, ZSTD_CCtx_params */

#ifdef ZSTD_CUDA
#  include <cuda_runtime.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum block size supported by the GPU match finder */
#define ZSTD_GPU_BLOCKSIZE_MAX ZSTD_BLOCKSIZE_MAX

/* Maximum number of sequences per block */
#define ZSTD_GPU_MAX_SEQUENCES 65536

/* Maximum match length supported */
#define ZSTD_GPU_MAXMATCH ((size_t)ZSTD_REPNUM * ZSTD_WINDOWMAX + MINMATCH)

/* Maximum number of 128 KB blocks processed per GPU batch */
#define ZSTD_GPU_MAX_BATCH 64u

/* Maximum sliding window size (1 MB) */
#define ZSTD_GPU_WINDOW_MAX (1u << 20)

/* ======================================================================== */
/* GPU Match Finder Types                                                     */
/* ======================================================================== */

/**
 * Single sequence entry from the GPU match finder.
 * position is used to sort results in correct order on the host.
 */
typedef struct {
    unsigned position;    /**< Position in input where match starts */
    unsigned offset;      /**< Match offset (distance to match) */
    unsigned matchLength; /**< Length of the match */
} ZSTD_GpuSequence;

/**
 * GPU match finder compression parameters.
 */
typedef struct {
    unsigned hashLog;       /**< Log of hash table size (6-30) */
    unsigned chainLog;      /**< Log of chain table size (6-30) */
    unsigned searchLog;     /**< Log of number of chain entries to search (1-29) */
    unsigned minMatch;      /**< Minimum match length (3-7) */
    unsigned strategy;      /**< Compression strategy: 1-9 (ZSTD_fast to ZSTD_btultra2) */
    unsigned windowLog;     /**< Log of the CCtx compression window */
} ZSTD_GpuCompressionParameters;

/**
 * GPU match finder context (opaque).
 */
typedef struct ZSTD_GpuMatchFinder_s ZSTD_GpuMatchFinder;

/**
 * Per-submitted-batch bookkeeping (double-buffered: batch j uses slot j%2).
 */
typedef struct {
    unsigned buf;        /**< staging/result buffer index (0/1) */
    unsigned count;      /**< number of blocks in the batch */
    unsigned seqsCopied; /**< the batch's sequence D2H has been issued */
} ZSTD_GpuBatchInfo;

/**
 * GPU match finder internal state (full definition for C file).
 */
struct ZSTD_GpuMatchFinder_s {
    ZSTD_GpuCompressionParameters params;
    unsigned windowSize;   /**< sliding window in bytes (capped at 1 MB) */
    unsigned hashLog;
    unsigned minMatch;
    unsigned mls;          /**< fingerprint length (== minMatch, clamped 4..8) */
    unsigned B;            /**< blocks per batch */

    /* Window / frame state (host-authoritative) */
    unsigned winBase;      /**< absolute position of d_window[0] */
    unsigned winUsed;      /**< valid bytes in d_window */
    unsigned framePos;     /**< absolute position of the next byte to append */

    /* Batch state */
    unsigned batchCount;             /**< blocks staged in the current batch */
    unsigned batchBuf;               /**< current staging buffer (0/1) */
    unsigned batchStagingOffset[ZSTD_GPU_MAX_BATCH + 1];  /**< per in-batch block, offset in staging (index count = total bytes) */
    unsigned batchSize[ZSTD_GPU_MAX_BATCH];
    unsigned batchSeqsCopied[2];
    unsigned batchIdx;               /**< number of batches submitted so far */
    ZSTD_GpuBatchInfo batchInfo[2];

    /* Host-side chunk pipeline (G19): the host stages a look-ahead of up to
     * 2*B blocks per frame chunk so the GPU batch pipeline runs ahead of the
     * host encoder (the GPU processes batch k+1 while the host encodes batch k).
     * Block indices below are global (within the frame). The next block to
     * stage is curBatchStart + batchCount (derived, not stored). */
    const void* chunkSrc;            /**< base of the current frame chunk */
    size_t      chunkSize;           /**< size of the current frame chunk */
    unsigned    chunkStartBlock;     /**< global index of the chunk's first block */
    unsigned    chunkEndBlock;       /**< global index one past the chunk's last block */
    unsigned    nextWait;            /**< global index of the next block to wait for */
    unsigned    curBatchStart;       /**< global index of the first block in the current (unsubmitted) batch */
    unsigned    histIdx;             /**< rotating index (0/1) for the submitted-batch history */
    unsigned    histBatchIdx[2];     /**< batchIdx of the last 2 submitted batches */
    unsigned    histStart[2];        /**< global index of the first block in the last 2 submitted batches */
    unsigned    histCount[2];        /**< number of blocks in the last 2 submitted batches */
    unsigned    inFlight;            /**< submitted batches not yet fully consumed (<= 2: the double-buffered pipeline) */
    int         chunkMode;           /**< 1 if the frame chunk loop drives the chunk pipeline */

    /* CUDA resources */
#ifdef ZSTD_CUDA
    unsigned char* d_window;          /**< device sliding window (windowSize + 128 KB) */
    unsigned char* d_windowTmp;       /**< temp buffer for the overlapping window slide */
    unsigned*      d_table;           /**< persistent fingerprint -> latest pos+1 */
    unsigned long long* d_keyA;       /**< sort-free: (absPos,hash) words grouped by hash row */
    unsigned long long* d_keyB;       /**< (retained; unused by the sort-free path) */
    unsigned*      d_slot;            /**< in-batch position -> index in the d_keyA row array */
    unsigned*      d_rowStart;        /**< per-hash row start in d_keyA (whole batch) */
    unsigned*      d_rowEnd;          /**< per-hash row end (whole batch) */
    /* Sort-free device search (G6): per-position scratch + per-hash row state.
     * d_posHash packs (absolute position, fingerprint) into one 64-bit word
     * ((absPos << 32) | hash) and is filled in one pass (gpuFillHist); the
     * scatter copies it into the row array (d_keyA), so the search reads the
     * predecessor's (absPos, hash) from a single 8-byte load and neither the
     * search nor the insert re-hashes the window or re-runs the block-locate
     * loop (the frame-block index is derived from the absolute position). d_hist
     * is the per-hash count; d_rowStart is its exclusive prefix (the row
     * starts); d_rowCounter is the scatter write pointer (zeroed per batch).
     * The d_blockTotal/d_blockPrefix and d_subTotal/d_subPrefix pairs are the
     * two-level exclusive scan of d_hist. */
    unsigned long long* d_posHash;    /**< per-position (absPos<<32)|hash [totalPos] */
    unsigned*      d_hist;            /**< per-hash position count [1<<hashLog] */
    unsigned*      d_rowCounter;      /**< scatter write pointers [1<<hashLog] */
    unsigned*      d_blockTotal;      /**< row-scan per-block totals [nScanBlocks] */
    unsigned*      d_blockPrefix;     /**< row-scan per-block exclusive prefix [nScanBlocks] */
    unsigned*      d_subTotal;        /**< row-scan sub-block totals [nSubBlocks] */
    unsigned*      d_subPrefix;       /**< row-scan sub-block exclusive prefix [nSubBlocks] */
    unsigned*      d_cand;            /**< per-position candidates [B][128K] x (off,len) */
    unsigned*      d_seq[2];          /**< final sequences [B][MAX_SEQS] x 12 B (double-buffered per batch parity: the next batch's greedy must not overwrite a batch whose D2H is still in flight) */
    unsigned*      d_nbSeq[2];        /**< per-block sequence count [B] (double-buffered, same reason) */
    unsigned*      d_nbCand;          /**< per-block candidate count [B] (0 => no match) */
    unsigned*      d_blockMeta;       /**< per-block (absStart, size) [B][2] */

    unsigned*      h_cand[2];         /**< pinned mirror of d_cand (double-buffered) */
    unsigned char* h_staging[2];      /**< pinned H2D staging (double-buffered) */
    unsigned*      h_nbSeq[2];        /**< pinned mirror of d_nbSeq */
    ZSTD_GpuSequence* h_seq[2];       /**< pinned mirror of d_seq (double-buffered) */

    cudaStream_t stream;              /**< compute stream */
    cudaStream_t copyStream;          /**< H2D/D2H stream */
    cudaEvent_t evH2d[2];
    cudaEvent_t evCompute[2];
    cudaEvent_t evD2H[2];
    cudaEvent_t evNbSeq[2];           /**< nbSeq D2H done (per buffer) */

    /* G17 per-phase profiling (ZSTD_GPU_PROFILE=1). Eight phases (G6
     * sort-free search replaced fillKeys/radix/recordSlot with
     * fillHist/rowScan/scatter) :
     * 0 H2D, 1 fillHist, 2 rowScan, 3 scatter, 4 search, 5 insert,
     * 6 greedy, 7 D2H. Times accumulate in ms. The events are double-buffered
     * by batch parity (G20.0) : with the chunk pipeline the next batch is
     * submitted before the previous batch's events are read, so a single set
     * would be overwritten. At most 2 batches are in flight, so parity is safe. */
    int         profile;
    double      phaseMs[8];
    unsigned    phaseCount;
    cudaEvent_t profEv[2][16];
#else
    void* d_window;
    void* d_windowTmp;
    void* d_keyA;
    void* d_keyB;
    void* d_slot;
    void* d_rowStart;
    void* d_rowEnd;
    void* d_posHash;
    void* d_hist;
    void* d_rowCounter;
    void* d_blockTotal;
    void* d_blockPrefix;
    void* d_subTotal;
    void* d_subPrefix;
    void* d_cand;
    void* d_nbCand;
    void* h_cand[2];
    void* h_staging[2];
    void* h_nbSeq[2];
    void* h_seq[2];
    void* stream;
    void* copyStream;
    void* evH2d[2];
    void* evCompute[2];
    void* evD2H[2];
    void* evNbSeq[2];
    void* profEv[2][16];
#endif
};

/* ======================================================================== */
/* Public API                                                                 */
/* ======================================================================== */

/**
 * Check if CUDA is available.
 * @return 1 if CUDA is available, 0 otherwise.
 */
int ZSTD_cudaAvailable(void);

/**
 * Create a GPU match finder context.
 * @param gpuMF[out] Pointer to receive the created context
 * @param cParams Compression parameters for the match finder
 * @param srcSize Expected source size (for memory pre-allocation)
 * @return 0 on success, error code on failure
 */
size_t ZSTD_createGpuMatchFinder(
    ZSTD_GpuMatchFinder** gpuMF,
    ZSTD_GpuCompressionParameters const* cParams,
    size_t srcSize);

/**
 * Free a GPU match finder context.
 * @param gpuMF The context to free (NULL is safe)
 * @return 0 on success
 */
size_t ZSTD_freeGpuMatchFinder(ZSTD_GpuMatchFinder* gpuMF);
size_t ZSTD_resetGpuMatchFinder(ZSTD_GpuMatchFinder* gpuMF);

/**
 * G20 : print the host-side per-block phase breakdown (total / gpuCall /
 * seqStore / stage / submit / wait) accumulated while ZSTD_GPU_HOSTPROF was
 * set. No-op otherwise. Called automatically from ZSTD_freeGpuMatchFinder.
 */
void ZSTD_gpuHostProfReport(void);

/* G20.2 : host-clock + phase accumulator for the .cu chunk path. No-ops
 * unless ZSTD_GPU_HOSTPROF is set. Phases: 3 stage, 4 submit, 5 wait. */
double ZSTD_gpuHostNow(void);
void ZSTD_gpuHostProfPhase(unsigned phase, double t0, double t1);

/**
 * Initialize the GPU match finder from CCtx parameters.
 * Called when GPU mode is first requested.
 */
size_t ZSTD_initGpuMatchFinderForCtx(ZSTD_CCtx* zc);

/**
 * Reset the window/table state for a new frame.
 */
size_t ZSTD_gpuMatchFinder_reset(ZSTD_GpuMatchFinder* gpuMF);

/**
 * Estimate GPU memory usage for a given configuration.
 * @return Estimated memory in bytes
 */
size_t ZSTD_estimateGpuMatchFinderMemory(
    ZSTD_GpuCompressionParameters const* cParams,
    size_t srcSize);

/* ---- Batch pipeline ---- */

/**
 * Stage a block into the current batch. Returns the in-batch index (0-based)
 * or a negative value on error.
 */
int ZSTD_gpuBatch_addBlock(ZSTD_GpuMatchFinder* gpuMF, void const* src, size_t srcSize);

/**
 * Submit the current batch to the GPU (asynchronous).
 */
size_t ZSTD_gpuBatch_submit(ZSTD_GpuMatchFinder* gpuMF);

/**
 * Wait for the result of block `blockInBatch` (within batch `batchIdx`) and
 * copy its final sequences into dst (up to dstCapacity). Returns the number
 * of sequences, or an error code.
 */
size_t ZSTD_gpuBatch_waitBlock(ZSTD_GpuMatchFinder* gpuMF, unsigned batchIdx,
                               unsigned blockInBatch, ZSTD_GpuSequence* dst,
                               size_t dstCapacity, void const* src, size_t srcSize);

/* ---- Chunk pipeline (G19) ---- */

/**
 * Begin a frame chunk for the chunk pipeline. Sets the chunk source so the
 * host can stage a look-ahead of blocks (up to 2*B) ahead of the block being
 * waited for. Must be called once per frame chunk, before any
 * ZSTD_gpuChunk_processBlock call. The global block index continues across
 * chunks within a frame (reset by ZSTD_gpuMatchFinder_reset).
 */
size_t ZSTD_gpuChunk_begin(ZSTD_GpuMatchFinder* gpuMF, void const* src, size_t srcSize);

/**
 * Stage the look-ahead and wait for the next block's sequences (in order).
 * Returns the number of sequences written to dst, or an error code.
 */
size_t ZSTD_gpuChunk_processBlock(ZSTD_GpuMatchFinder* gpuMF, ZSTD_GpuSequence* dst,
                                  size_t dstCapacity);

/**
 * Perform GPU-accelerated match finding for a single block (synchronous).
 * @return Number of sequences written, or error code
 */
/* The GPU path never emits repcodes : both rep arrays are passed through
 * unchanged (all zero). */
size_t ZSTD_compressBlock_gpu(
    ZSTD_GpuMatchFinder* gpuMF,
    ZSTD_Sequence* dst,
    size_t dstCapacity,
    void const* src,
    size_t srcSize,
    unsigned rep[ZSTD_REP_NUM],
    unsigned repOut[ZSTD_REP_NUM]);

/**
 * Check if GPU match finding should be used for this context.
 * @return 1 if GPU should be used, 0 otherwise.
 */
int ZSTD_shouldUseGpu(ZSTD_CCtx const* zc);

/**
 * GPU-accelerated version of ZSTD_buildSeqStore.
 */
size_t ZSTD_buildSeqStore_gpu(ZSTD_CCtx* zc, void const* src, size_t srcSize);


#ifdef __cplusplus
}
#endif

#endif /* ZSTD_CUDA_MATCHFINDER_H */
