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

/* Maximum hash table size (2^30 entries) */
#define ZSTD_GPU_HASHLOG_MAX 30

/* Maximum number of sequences per block */
#define ZSTD_GPU_MAX_SEQUENCES 65536

/* Maximum match length supported */
#define ZSTD_GPU_MAXMATCH ((size_t)ZSTD_REPNUM * ZSTD_WINDOWMAX + MINMATCH)

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
 * Result from the GPU match finder.
 */
typedef struct {
    unsigned nbSeq;
    ZSTD_GpuSequence sequences[ZSTD_GPU_MAX_SEQUENCES];
} ZSTD_GpuMatchResult;

/**
 * GPU match finder compression parameters.
 */
typedef struct {
    unsigned hashLog;       /**< Log of hash table size (6-30) */
    unsigned chainLog;      /**< Log of chain table size (6-30) */
    unsigned searchLog;     /**< Log of number of chain entries to search (1-29) */
    unsigned minMatch;      /**< Minimum match length (3-7) */
    unsigned strategy;      /**< Compression strategy: 1-9 (ZSTD_fast to ZSTD_btultra2) */
} ZSTD_GpuCompressionParameters;

/**
 * GPU match finder context (opaque).
 */
typedef struct ZSTD_GpuMatchFinder_s ZSTD_GpuMatchFinder;

/**
 * GPU match finder internal state (full definition for C file).
 */
struct ZSTD_GpuMatchFinder_s {
    ZSTD_GpuCompressionParameters params;
    size_t windowSize;
    size_t hashTableSize;
    int initialized;

    /* CUDA resources */
#ifdef ZSTD_CUDA
    unsigned char* d_window;          /**< Device pointer to sliding window */
    void* d_keyA;                     /**< Composite sort keys (fill buffer) */
    void* d_keyB;                     /**< Sort ping-pong buffer (final result after odd passes) */
    void* d_slot;                     /**< position -> index in sorted array */
    unsigned* d_rowStart;             /**< CSR row start index per hash */
    unsigned* d_rowEnd;               /**< CSR row end index per hash */
    void* d_cand;                     /**< Device candidates [counter, per-position...] */
    void* d_blockHist;                /**< Radix per-block histogram (512x256x4 B) */
    void* d_digitStart;               /**< Radix per-digit global starts (256x4 B) */
    void* cand_host;                  /**< Pinned host mirror of d_cand */
    cudaStream_t stream;              /**< CUDA stream handle */
#else
    void* d_window;                   /**< Placeholder for CUDA pointer */
    void* d_keyA;
    void* d_keyB;
    void* d_slot;
    void* d_rowStart;
    void* d_rowEnd;
    void* d_cand;
    void* d_blockHist;
    void* d_digitStart;
    void* cand_host;
    void* stream;
#endif

    /* Host-side buffers */
    ZSTD_GpuMatchResult* h_results;   /**< Host results buffer (pinned memory via cudaMallocHost) */
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
 * Estimate GPU memory usage for a given configuration.
 * @param cParams Compression parameters
 * @param srcSize Expected source size
 * @return Estimated memory in bytes
 */
size_t ZSTD_estimateGpuMatchFinderMemory(
    ZSTD_GpuCompressionParameters const* cParams,
    size_t srcSize);

/**
 * Perform GPU-accelerated match finding.
 * @param gpuMF The GPU match finder context
 * @param dst[out] Output buffer for results
 * @param dstCapacity Capacity of the output buffer
 * @param src Input data to compress
 * @param srcSize Size of the input data
 * @param rep[in,out] Repetition codes (updated by this function)
 * @param prevRep Repetition codes from the previous block
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
 * @param zc The zstd compression context
 * @return 1 if GPU should be used, 0 otherwise
 */
int ZSTD_shouldUseGpu(ZSTD_CCtx const* zc);

/**
 * GPU-accelerated version of ZSTD_buildSeqStore.
 * Replaces the CPU match finding with GPU match finding.
 */
size_t ZSTD_buildSeqStore_gpu(ZSTD_CCtx* zc, void const* src, size_t srcSize);


#ifdef __cplusplus
}
#endif

#endif /* ZSTD_CUDA_MATCHFINDER_H */
