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
 * This file contains the CUDA device kernels and host-side implementation
 * for GPU-accelerated hash chain match finding.
 *
 * Architecture:
 * - Each thread block processes one "row" of input (128 bytes)
 * - Each thread within a block processes one input position
 * - Hash tables are stored in constant memory for fast read access
 * - Chain tables are stored in global memory with caching
 * - The sliding window is stored in global memory
 *
 * The implementation follows the hash chain match finding algorithm:
 * 1. Hash 4 bytes of input to get a hash value
 * 2. Look up the hash in the hash table to get a candidate match
 * 3. Traverse the chain to find all candidates within the window
 * 4. Compare each candidate to find the longest match
 * 5. Optionally perform lazy search (depth 1, 2) for better matches
 */

#include "zstd_cuda_matchfinder.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/* Stub for ERR_getErrorString - GPU code doesn't need full error strings */
const char* ERR_getErrorString(const ERR_enum code)
{
    (void) code;
    return "GPU error";
}

/* ======================================================================== */
/* CUDA Device Functions                                                      */
/* ======================================================================== */

namespace
{
/**
 * Read 4 bytes from a pointer (device-safe version).
 */
__device__ __forceinline__
unsigned ZSTD_gpu_read32(void const* ptr)
{
    unsigned val;
    memcpy(&val, ptr, 4);
    return val;
}
}

/**
 * Host-side hash function for 4 bytes at a given pointer.
 * Uses a simple multiplicative hash (same as device version).
 */
static __host__ unsigned ZSTD_gpu_hashPtr_host(void const* ptr, const unsigned hashLog)
{
    unsigned h;
    memcpy(&h, ptr, 4);
    return (h * 2654435761U) >> (32 - hashLog);
}

namespace
{
/**
 * Hash function for 4 bytes at a given pointer.
 * Uses a simple multiplicative hash.
 */
__device__ __forceinline__
unsigned ZSTD_gpu_hashPtr(void const* ptr, const unsigned hashLog)
{
    unsigned const h = ZSTD_gpu_read32(ptr);
    return (h * 2654435761U) >> (32 - hashLog);
}
}

namespace
{
/**
 * Count matching bytes between two pointers.
 * Returns the length of the match.
 *
 * @param ip Pointer to current position
 * @param match Pointer to match candidate
 * @param iend End of input buffer
 * @return Length of the match (minimum 0)
 */
__device__ __forceinline__
size_t ZSTD_gpu_count(void const* ip, void const* match, void const* iend)
{
    /* Calculate max match length from srcSize and position */
    const auto srcSize = static_cast<unsigned>(static_cast<unsigned char const*>(iend) - static_cast<unsigned char const
                                                   *>(ip));
    size_t length = 0;

    /* Compare 4 bytes at a time using offset-based access to avoid
     * pointer arithmetic issues with mixed byte/unsigned pointers */
    while (length + 4 <= srcSize)
    {
        unsigned val1, val2;
        memcpy(&val1, static_cast<unsigned char const*>(ip) + length, 4);
        memcpy(&val2, static_cast<unsigned char const*>(match) + length, 4);
        if (const unsigned diff = val1 ^ val2)
        {
            /* Found mismatch, count trailing zeros */
            length += (__ffs(diff) - 1) / 8;
            return length;
        }
        length += 4;
    }

    /* Handle remaining bytes */
    while (length < srcSize && static_cast<unsigned char const*>(ip)[length] == static_cast<unsigned char const*>(match)
           [length])
        length++;

    return length;
}
}

/**
 * Sequential GPU kernel for hash chain match finding.
 *
 * This kernel performs a single-threaded sequential pass through the input,
 * exactly like zstd's CPU fast match finder. This ensures that:
 * 1. Only non-overlapping sequences are emitted
 * 2. Literal lengths are computed correctly
 * 3. Each sequence's position is the start of the match, and
 *    litLength = position - previous_position
 *
 * Algorithm:
 * 1. Start at position 0
 * 2. At each position, compute hash and look up in hash table
 * 3. If a match is found, compute match length by comparing bytes
 * 4. If match length >= minMatch, emit a sequence and advance past the match
 * 5. Otherwise, advance to the next position
 * 6. Repeat until we reach the end of input
 */
__global__
static void ZSTD_gpu_hashChainMatchFinder(unsigned char const* d_window, const unsigned* d_hashTable,
                                          const unsigned* d_chainTable, ZSTD_GpuMatchResult* d_results,
                                          const unsigned hashLog, const unsigned chainLog, const unsigned searchLog,
                                          const unsigned minMatch, const size_t srcSize, const unsigned maxDistance)
{
    /* Only use thread 0 - this is a sequential kernel */
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    unsigned const chainMask = (1u << chainLog) - 1;
    unsigned const maxDist = 1u << maxDistance;

    /* State variables - sequential traversal */
    unsigned pos = 0;
    unsigned nbSeq = 0;

    while (pos + minMatch <= srcSize && nbSeq < ZSTD_GPU_MAX_SEQUENCES)
    {
        /* Compute hash at current position */
        unsigned h = ZSTD_gpu_hashPtr(d_window + pos, hashLog);

        /* Look up candidate match in hash table */
        unsigned matchIndex = d_hashTable[h];

        /* Find the best match by traversing the chain */
        size_t bestMatchLen = 0;
        unsigned bestMatchOffset = 0;
        unsigned attempts = 1u << searchLog;

        while (matchIndex < pos && attempts > 0)
        {
            /* Skip if match is outside window */
            if (pos - matchIndex > maxDist)
            {
                matchIndex = d_chainTable[matchIndex & chainMask];
                attempts--;
                continue;
            }

            /* Skip sentinel value */
            if (matchIndex == 0xFFFFFFFFu)
            {
                matchIndex = d_chainTable[matchIndex & chainMask];
                attempts--;
                continue;
            }

            /* Quick 4-byte check */
            if (pos + 4 <= srcSize && matchIndex + 4 <= srcSize &&
                ZSTD_gpu_read32(d_window + pos) == ZSTD_gpu_read32(d_window + matchIndex))
            {
                size_t currentMl = ZSTD_gpu_count(d_window + pos, d_window + matchIndex, d_window + srcSize);

                if (currentMl > bestMatchLen)
                {
                    bestMatchLen = currentMl;
                    bestMatchOffset = pos - matchIndex;
                }
            }

            matchIndex = d_chainTable[matchIndex & chainMask];
            attempts--;
        }

        /* If we found a good match, emit a sequence (tables pre-populated on host) */
        if (bestMatchLen >= minMatch)
        {
            d_results->sequences[nbSeq].position = pos;
            d_results->sequences[nbSeq].offset = bestMatchOffset;
            d_results->sequences[nbSeq].matchLength = (unsigned)bestMatchLen;
            nbSeq++;

            /* Advance past the match */
            pos += (unsigned)bestMatchLen;
        }
        else
        {
            /* No good match, advance by one */
            pos++;
        }
    }

    /* Store the count */
    d_results->nbSeq = nbSeq;
}

/* ======================================================================== */
/* Host-Side Implementation                                                   */
/* ======================================================================== */
/* Note: ZSTD_GpuMatchFinder_s is fully defined in zstd_cuda_matchfinder.h */

/**
 * Initialize the GPU match finder context.
 */
// static size_t ZSTD_initGpuMatchFinder(ZSTD_GpuMatchFinder* gpuMF)
// {
//     if (!gpuMF || !gpuMF->initialized) return ERROR(GPU_matchFinder_notInitialized);
//
//     /* Allocate hash table on GPU and zero it */
//     size_t const hashTableBytes = (size_t) 1 << gpuMF->params.hashLog;
//     cudaError_t err = cudaMalloc((void**) &gpuMF->d_hashTable, hashTableBytes * sizeof(unsigned));
//     if (err != cudaSuccess)
//     {
//         fprintf(stderr, "DEBUG: cudaMalloc hashTable failed: %s (code=%d)\n", cudaGetErrorString(err), (int) err);
//         return ERROR(GPU_cudaAllocationFailed);
//     }
//     err = cudaMemset(gpuMF->d_hashTable, 0xFF, hashTableBytes * sizeof(unsigned));
//     if (err != cudaSuccess)
//     {
//         cudaFree(gpuMF->d_hashTable);
//         gpuMF->d_hashTable = NULL;
//         return ERROR(GPU_cudaAllocationFailed);
//     }
//
//     /* Allocate chain table on GPU and zero it */
//     size_t const chainTableBytes = (size_t) 1 << gpuMF->params.chainLog;
//     err = cudaMalloc((void**) &gpuMF->d_chainTable, chainTableBytes * sizeof(unsigned));
//     if (err != cudaSuccess)
//     {
//         cudaFree(gpuMF->d_hashTable);
//         gpuMF->d_hashTable = NULL;
//         return ERROR(GPU_cudaAllocationFailed);
//     }
//     err = cudaMemset(gpuMF->d_chainTable, 0xFF, chainTableBytes * sizeof(unsigned));
//     if (err != cudaSuccess)
//     {
//         cudaFree(gpuMF->d_chainTable);
//         cudaFree(gpuMF->d_hashTable);
//         gpuMF->d_chainTable = NULL;
//         gpuMF->d_hashTable = NULL;
//         return ERROR(GPU_cudaAllocationFailed);
//     }
//
//     /* Allocate results buffer on GPU and zero it */
//     err = cudaMalloc((void**) &gpuMF->d_results, sizeof(ZSTD_GpuMatchResult));
//     if (err != cudaSuccess)
//     {
//         cudaFree(gpuMF->d_chainTable);
//         cudaFree(gpuMF->d_hashTable);
//         gpuMF->d_chainTable = NULL;
//         gpuMF->d_hashTable = NULL;
//         return ERROR(GPU_cudaAllocationFailed);
//     }
//     err = cudaMemset(gpuMF->d_results, 0, sizeof(ZSTD_GpuMatchResult));
//     if (err != cudaSuccess)
//     {
//         cudaFree(gpuMF->d_results);
//         cudaFree(gpuMF->d_chainTable);
//         cudaFree(gpuMF->d_hashTable);
//         gpuMF->d_results = NULL;
//         gpuMF->d_chainTable = NULL;
//         gpuMF->d_hashTable = NULL;
//         return ERROR(GPU_cudaAllocationFailed);
//     }
//
//     return 0;
// }

/**
 * Update the hash and chain tables on the GPU.
 *
 * This function copies the current state of the hash and chain tables
 * from the CPU match state to the GPU.
 */
// static size_t ZSTD_updateGpuTables(
//     ZSTD_GpuMatchFinder* gpuMF,
//     void const* src,
//     size_t srcSize,
//     unsigned nextToUpdate)
// {
//     if (!gpuMF || !src)
//     {
//         fprintf(stderr, "DEBUG: ZSTD_updateGpuTables invalid parameter\n");
//         return ERROR(GPU_invalidParameter);
//     }
//     fprintf(stderr, "DEBUG: ZSTD_updateGpuTables srcSize=%zu\n", srcSize);
//
//     unsigned char const* const ip = (unsigned char const*) src;
//     unsigned const hashLog = gpuMF->params.hashLog;
//     unsigned const chainLog = gpuMF->params.chainLog;
//     unsigned const chainMask = (1u << chainLog) - 1;
//
//     /* Copy input data to GPU window */
//     cudaError_t err = cudaMemcpyAsync(
//         gpuMF->d_window,
//         src,
//         srcSize,
//         cudaMemcpyHostToDevice,
//         gpuMF->stream);
//     if (err != cudaSuccess)
//     {
//         fprintf(stderr, "DEBUG: ZSTD_updateGpuTables cudaMemcpy window failed: %s (code=%d)\n", cudaGetErrorString(err),
//                 (int) err);
//         return ERROR(GPU_cudaMemcpyFailed);
//     }
//
//     /* Update hash and chain tables on GPU */

/**
 * Run the GPU match finding kernel.
 */
static size_t ZSTD_runGpuMatchFinding(
    ZSTD_GpuMatchFinder* gpuMF,
    void const* src,
    const size_t srcSize)
{
    if (!gpuMF || !src) return ERROR(GPU_invalidParameter);

    unsigned const hashLog = gpuMF->params.hashLog;
    unsigned const chainLog = gpuMF->params.chainLog;
    unsigned const searchLog = gpuMF->params.searchLog;
    unsigned const minMatch = gpuMF->params.minMatch;
    /* Compute window log from srcSize: use enough bits to cover srcSize */
    /* In zstd, windowLog determines max match distance. For a single block,
     * we need at least log2(srcSize) to find all matches within the block. */
    unsigned windowLog = 15; /* default 32KB */
    {
        size_t s = srcSize;
        while (s > 1)
        {
            s >>= 1;
            windowLog++;
        }
        if (windowLog > 27) windowLog = 27; /* cap at zstd max */
        if (windowLog < 15) windowLog = 15;
    }
    unsigned const maxDistance = windowLog;

    /* Populate hash and chain tables ENTIRELY on host, then copy to device
     * in single operations. This is much faster and avoids per-element
     * cudaMemcpy synchronization issues. */
    size_t const hashTableSize = static_cast<size_t>(1) << hashLog;
    size_t const chainTableSize = static_cast<size_t>(1) << chainLog;

    /* Allocate host-side buffers for tables */
    const auto h_hashTable = static_cast<unsigned*>(malloc(hashTableSize * sizeof(unsigned)));
    auto* h_chainTable = static_cast<unsigned*>(malloc(chainTableSize * sizeof(unsigned)));
    if (!h_hashTable || !h_chainTable)
    {
        free(h_hashTable);
        free(h_chainTable);
        return ERROR(GPU_allocationFailed);
    }

    /* Initialize hash table to 0xFFFFFFFF (sentinel = no entry yet)
     * This ensures chain_table[pos] = 0xFFFFFFFF for the first position
     * with a given hash, preventing self-loops in the chain traversal. */
    memset(h_hashTable, 0xFF, hashTableSize * sizeof(unsigned));
    memset(h_chainTable, 0xFF, chainTableSize * sizeof(unsigned));

    /* Build hash and chain tables on host (sequential, correct order) */
    {
        auto const* const ip = static_cast<unsigned char const*>(src);
        auto const chainMask = static_cast<unsigned>(chainTableSize - 1);
        for (size_t pos = 0; pos + 4 <= srcSize; pos++)
        {
            unsigned const h = ZSTD_gpu_hashPtr_host(ip + pos, hashLog);
            const unsigned prevInChain = h_hashTable[h];
            h_chainTable[pos & chainMask] = prevInChain;
            h_hashTable[h] = static_cast<unsigned>(pos);
        }
    }

    /* Copy tables to device in single operations */
    cudaError_t err = cudaMemcpyAsync(gpuMF->d_hashTable, h_hashTable, hashTableSize * sizeof(unsigned),
                                      cudaMemcpyHostToDevice, gpuMF->stream);
    if (err != cudaSuccess)
    {
        free(h_hashTable);
        free(h_chainTable);
        return ERROR(GPU_cudaMemcpyFailed);
    }

    err = cudaMemcpyAsync(gpuMF->d_chainTable, h_chainTable, chainTableSize * sizeof(unsigned), cudaMemcpyHostToDevice,
                          gpuMF->stream);
    if (err != cudaSuccess)
    {
        free(h_hashTable);
        free(h_chainTable);
        return ERROR(GPU_cudaMemcpyFailed);
    }

    free(h_hashTable);
    free(h_chainTable);

    /* Synchronize gpuMF->stream to ensure the input data copy (d_window)
     * from ZSTD_compressBlock_gpu is complete before the kernel reads it. */
    cudaStreamSynchronize(gpuMF->stream);

    /* Initialize results buffer on device BEFORE launching kernel
     * to avoid race condition with atomicAdd in kernel */
    err = cudaMemsetAsync(gpuMF->d_results, 0, sizeof(ZSTD_GpuMatchResult), gpuMF->stream);
    if (err != cudaSuccess)
        return ERROR(GPU_cudaMemcpyFailed);

    /* Now launch the match finding kernel */

    ZSTD_gpu_hashChainMatchFinder<<<1, 1, 0, gpuMF->stream>>>(
        gpuMF->d_window, gpuMF->d_hashTable, gpuMF->d_chainTable, gpuMF->d_results, hashLog, chainLog, searchLog,
        minMatch, srcSize, maxDistance);

    /* Check for kernel launch errors */
    if (cudaGetLastError() != cudaSuccess)
        return ERROR(GPU_kernelLaunchFailed);

    /* Synchronize stream to ensure kernel is complete before copying results 
    if (const cudaError_t syncErr = cudaStreamSynchronize(gpuMF->stream); syncErr != cudaSuccess)
    {
        fprintf(stderr, "DEBUG: Stream sync error before copy: %s (code=%d)\n", cudaGetErrorString(syncErr),
                static_cast<int>(syncErr));
        return ERROR(GPU_cudaMemcpyFailed);
    }

    /* Copy results back using synchronous cudaMemcpy */
    if (const cudaError_t copyErr = cudaMemcpy(gpuMF->h_results, gpuMF->d_results, sizeof(ZSTD_GpuMatchResult),
                                               cudaMemcpyDeviceToHost); copyErr != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy results failed: %s\n", cudaGetErrorString(copyErr));
        return ERROR(GPU_cudaMemcpyFailed);
    }

    return 0;
}

/**
 * Compare function for sorting sequences by position (for qsort).
 */
static int compareSequences(const void* a, const void* b)
{
    const auto seqA = static_cast<const ZSTD_GpuSequence*>(a);
    const auto seqB = static_cast<const ZSTD_GpuSequence*>(b);
    if (seqA->position < seqB->position) return -1;
    if (seqA->position > seqB->position) return 1;
    return 0;
}

/**
 * Convert GPU match results to ZSTD_Sequence format.
 *
 * The GPU kernel stores results with position information.
 * We sort them by position and calculate literal lengths correctly.
 */
static size_t ZSTD_convertResultsToSequences(ZSTD_GpuMatchFinder const* gpuMF, ZSTD_Sequence* dst,
                                             const size_t dstCapacity, void const* src, unsigned rep[])
{
    if (!gpuMF || !dst || !src) return ERROR(GPU_invalidParameter);

    unsigned seqIdx = 0;
    size_t currentPos = 0; /* Track current position in input */

    /* Copy results from host buffer */
    ZSTD_GpuMatchResult const* results = gpuMF->h_results;

    /* Sort results by position so they're in input order */
    if (results->nbSeq > 1)
        qsort((void*) results->sequences, results->nbSeq, sizeof(ZSTD_GpuSequence), compareSequences);

    /* Convert sequences in position order */
    for (unsigned i = 0; i < results->nbSeq && seqIdx < dstCapacity; i++)
    {
        unsigned const position = results->sequences[i].position;
        unsigned const offset = results->sequences[i].offset;
        unsigned const matchLen = results->sequences[i].matchLength;

        if (offset == 0) continue; /* No match */

        /* Calculate literal length: bytes from current position to match start */
        const auto litLen = static_cast<unsigned>(position - currentPos);

        /* Store sequence */
        dst[seqIdx].litLength = litLen;
        dst[seqIdx].matchLength = matchLen; /* Actual match length - wrapper converts to mlBase */
        dst[seqIdx].offset = offset;
        dst[seqIdx].rep = 0; /* Repcodes not yet supported */

        seqIdx++;
        currentPos = position + matchLen; /* Advance to end of this match */
    }

    /* Update repetition codes */
    if (results->nbSeq > 0)
    {
        rep[1] = rep[0];
        rep[0] = results->sequences[results->nbSeq - 1].offset;
    }

    return seqIdx;
}

/* ======================================================================== */
/* Public API Implementation                                                  */
/* ======================================================================== */

size_t ZSTD_createGpuMatchFinder(
    ZSTD_GpuMatchFinder** gpuMF,
    ZSTD_GpuCompressionParameters const* cParams,
    const size_t srcSize)
{
    if (!gpuMF || !cParams) return ERROR(GPU_invalidParameter);

    /* Validate parameters */
    if (cParams->hashLog < 6 || cParams->hashLog > 30) return ERROR(GPU_invalidHashLog);
    if (cParams->chainLog < 6 || cParams->chainLog > 30) return ERROR(GPU_invalidChainLog);
    if (cParams->searchLog < 1 || cParams->searchLog > 29) return ERROR(GPU_invalidSearchLog);
    if (cParams->minMatch < 3 || cParams->minMatch > 7) return ERROR(GPU_invalidMinMatch);
    /* Allocate context */
    const auto ctx = static_cast<ZSTD_GpuMatchFinder*>(malloc(sizeof(ZSTD_GpuMatchFinder)));
    if (!ctx) return ERROR(GPU_allocationFailed);

    memset(ctx, 0, sizeof(ZSTD_GpuMatchFinder));
    ctx->params = *cParams;
    ctx->windowSize = srcSize + ZSTD_BLOCKSIZE_MAX;
    ctx->initialized = 1;

    /* Create CUDA stream */
    cudaError_t err = cudaStreamCreate(&ctx->stream);
    if (err != cudaSuccess)
    {
        free(ctx);
        return ERROR(GPU_cudaStreamCreationFailed);
    }

    /* Allocate h_results using cudaMallocHost (pinned memory) so that
     * cudaMemcpyAsync can reliably transfer results from device to host.
     * Regular malloc memory is pageable and may cause "misaligned address"
     * errors when used as a cudaMemcpy destination. */
    err = cudaMallocHost(reinterpret_cast<void**>(&ctx->h_results), sizeof(ZSTD_GpuMatchResult));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "DEBUG: cudaMallocHost h_results failed: %s (code=%d)\n", cudaGetErrorString(err),
                static_cast<int>(err));
        cudaStreamDestroy(ctx->stream);
        free(ctx);
        return ERROR(GPU_cudaAllocationFailed);
    }

    /* Allocate GPU memory */
    size_t const windowBytes = ctx->windowSize;
    err = cudaMalloc(reinterpret_cast<void**>(&ctx->d_window), windowBytes);
    if (err != cudaSuccess)
    {
        cudaStreamDestroy(ctx->stream);
        free(ctx);
        return ERROR(GPU_cudaAllocationFailed);
    }

    ctx->hashTableSize = static_cast<size_t>(1) << cParams->hashLog;
    ctx->chainTableSize = static_cast<size_t>(1) << cParams->chainLog;

    err = cudaMalloc(reinterpret_cast<void**>(&ctx->d_hashTable), ctx->hashTableSize * sizeof(unsigned));
    if (err != cudaSuccess)
    {
        cudaFree(ctx->d_window);
        cudaStreamDestroy(ctx->stream);
        free(ctx);
        return ERROR(GPU_cudaAllocationFailed);
    }

    err = cudaMalloc(reinterpret_cast<void**>(&ctx->d_chainTable), ctx->chainTableSize * sizeof(unsigned));
    if (err != cudaSuccess)
    {
        cudaFree(ctx->d_hashTable);
        cudaFree(ctx->d_window);
        cudaStreamDestroy(ctx->stream);
        free(ctx);
        return ERROR(GPU_cudaAllocationFailed);
    }

    err = cudaMalloc(reinterpret_cast<void**>(&ctx->d_results), sizeof(ZSTD_GpuMatchResult));
    if (err != cudaSuccess)
    {
        cudaFree(ctx->d_chainTable);
        cudaFree(ctx->d_hashTable);
        cudaFree(ctx->d_window);
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

    if (gpuMF->initialized)
    {
        if (gpuMF->d_window) cudaFree(gpuMF->d_window);
        if (gpuMF->d_hashTable) cudaFree(gpuMF->d_hashTable);
        if (gpuMF->d_chainTable) cudaFree(gpuMF->d_chainTable);
        if (gpuMF->d_results) cudaFree(gpuMF->d_results);
        if (gpuMF->h_results) cudaFreeHost(gpuMF->h_results);
        if (gpuMF->stream) cudaStreamDestroy(gpuMF->stream);
    }

    free(gpuMF);
    return 0;
}

size_t ZSTD_compressBlock_gpu(ZSTD_GpuMatchFinder* gpuMF, ZSTD_Sequence* dst, const size_t dstCapacity, void const* src,
                              const size_t srcSize, unsigned rep[ZSTD_REP_NUM], unsigned const prevRep[ZSTD_REP_NUM])
{
    if (!gpuMF || !src || !dst) return ERROR(GPU_invalidParameter);
    if (srcSize > ZSTD_GPU_BLOCKSIZE_MAX) return ERROR(GPU_blockTooLarge);

    /* Copy input data to GPU */
    if (const cudaError_t err = cudaMemcpyAsync(gpuMF->d_window, src, srcSize, cudaMemcpyHostToDevice, gpuMF->stream);
        err != cudaSuccess)
        return ERROR(GPU_cudaMemcpyFailed);

    /* Run match finding kernel */
    if (const size_t result = ZSTD_runGpuMatchFinding(gpuMF, src, srcSize))
        return result;

    /* Convert results to ZSTD_Sequence format */
    size_t const nbSeq = ZSTD_convertResultsToSequences(gpuMF, dst, dstCapacity, src, rep);

    return nbSeq;
}

size_t ZSTD_estimateGpuMatchFinderMemory(ZSTD_GpuCompressionParameters const* cParams, const size_t srcSize)
{
    if (!cParams) return 0;

    size_t total = 0;

    /* Window buffer */
    total += srcSize + ZSTD_BLOCKSIZE_MAX;

    /* Hash table */
    total += static_cast<size_t>(1) << cParams->hashLog;
    total *= sizeof(unsigned);

    /* Chain table */
    total += static_cast<size_t>(1) << cParams->chainLog;
    total *= sizeof(unsigned);

    /* Results buffer */
    total += sizeof(ZSTD_GpuMatchResult);

    return total;
}

int ZSTD_cudaAvailable()
{
    auto deviceCount = 0;
    return cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0 ? 1 : 0;
}
