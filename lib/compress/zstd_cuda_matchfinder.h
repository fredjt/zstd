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
    unsigned posBits;      /**< bits for the in-batch position field */
    unsigned nPasses;      /**< radix passes over the hash bits */
    unsigned nBlocksMax;   /**< max CUDA blocks per batch (B*512) */

    /* Window / frame state (host-authoritative) */
    unsigned winBase;      /**< absolute position of d_window[0] */
    unsigned winUsed;      /**< valid bytes in d_window */
    unsigned framePos;     /**< absolute position of the next byte to append */

    /* Batch state */
    unsigned batchCount;             /**< blocks staged in the current batch */
    unsigned batchBuf;               /**< current staging buffer (0/1) */
    unsigned batchStagingOffset[2];  /**< per in-batch block, offset in staging */
    unsigned batchSize[ZSTD_GPU_MAX_BATCH];
    unsigned batchSeqsCopied[2];
    unsigned batchIdx;               /**< number of batches submitted so far */
    ZSTD_GpuBatchInfo batchInfo[2];

    /* CUDA resources */
#ifdef ZSTD_CUDA
    unsigned char* d_window;          /**< device sliding window (windowSize + 128 KB) */
    unsigned char* d_windowTmp;       /**< temp buffer for the overlapping window slide */
    unsigned*      d_table;           /**< persistent fingerprint -> latest pos+1 */
    unsigned long long* d_keyA;       /**< sort keys (fill buffer) */
    unsigned long long* d_keyB;       /**< sort ping-pong buffer */
    unsigned*      d_slot;            /**< in-batch position -> sorted index */
    unsigned*      d_rowStart;        /**< per-hash row start (whole batch) */
    unsigned*      d_rowEnd;          /**< per-hash row end (whole batch) */
    unsigned*      d_blockHist;       /**< radix per-CUDA-block histogram [nBlocks][256] */
    unsigned*      d_blockStartLocal; /**< radix within-group exclusive prefix [nBlocks][256] */
    unsigned*      d_blockCounter;    /**< radix scatter counters [nBlocks][256] */
    unsigned*      d_groupTotal;      /**< radix per-group digit totals [nGroups][256] */
    unsigned*      d_groupPrefix;     /**< radix group prefix [nGroups][256] */
    unsigned*      d_digitTotal;      /**< radix digit totals [256] */
    unsigned*      d_digitStart;      /**< radix digit global starts [256] */
    unsigned*      d_cand;            /**< per-position candidates [B][128K] x (off,len) */
    unsigned*      d_seq;             /**< final sequences [B][MAX_SEQS] x 12 B */
    unsigned*      d_nbSeq;           /**< per-block sequence count [B] */
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

    /* G17 per-phase profiling (ZSTD_GPU_PROFILE=1). Nine phases:
     * 0 H2D, 1 fillKeys, 2 radix, 3 recordSlot, 4 rowBounds, 5 search,
     * 6 insert, 7 greedy, 8 D2H. Times accumulate in ms. */
    int         profile;
    double      phaseMs[9];
    unsigned    phaseCount;
    cudaEvent_t profEv[18];
#else
    void* d_window;
    void* d_windowTmp;
    void* d_keyA;
    void* d_keyB;
    void* d_slot;
    void* d_rowStart;
    void* d_rowEnd;
    void* d_cand;
    void* d_nbCand;
    void* d_blockHist;
    void* d_digitStart;
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
