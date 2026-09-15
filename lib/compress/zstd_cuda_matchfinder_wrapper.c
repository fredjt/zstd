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
 * @file zstd_cuda_matchfinder_wrapper.c
 * @brief C wrapper for GPU-accelerated match finding in zstd.
 *
 * This file provides the C interface that bridges the CUDA implementation
 * with zstd\'s internal compression flow. It handles:
 * - GPU context management (creation, initialization, destruction)
 * - Integration with ZSTD_buildSeqStore for GPU-accelerated compression
 */

#include <stdio.h>

#include "zstd_compress_internal.h"
#include "zstd_cuda_matchfinder.h"

/* ======================================================================== */
/* Local definitions for types not exposed in public headers                 */
/* ======================================================================== */

/* ZSTDbss_compress/ZSTDbss_noCompress are local typedefs in zstd_compress.c */
typedef enum { ZSTDbss_compress, ZSTDbss_noCompress } ZSTD_BuildSeqStore_e;

/* Inline version of ZSTD_storeLastLiterals (static in zstd_compress.c) */
static void ZSTD_storeLastLiterals_gpu(SeqStore_t* seqStorePtr, const BYTE* anchor, size_t lastLLSize)
{
    ZSTD_memcpy(seqStorePtr->lit, anchor, lastLLSize);
    seqStorePtr->lit += lastLLSize;
}

/* ======================================================================== */
/* GPU Match Finder Context Management                                       */
/* ======================================================================== */

/**
 * Initialize the GPU match finder from CCtx parameters.
 * Called when GPU mode is first requested.
 */
size_t ZSTD_initGpuMatchFinderForCtx(ZSTD_CCtx* zc)
{
    ZSTD_GpuCompressionParameters gpuParams = {0};

    if (!zc) return ERROR(GPU_invalidParameter);

    /* Use ZSTD_BLOCKSIZE_MAX as the default block size */
    constexpr size_t srcSize = ZSTD_BLOCKSIZE_MAX;

    /* Configure GPU parameters from CCtx params */
    gpuParams.hashLog = zc->appliedParams.cParams.hashLog;
    gpuParams.chainLog = zc->appliedParams.cParams.chainLog;
    gpuParams.searchLog = zc->appliedParams.cParams.searchLog;
    gpuParams.minMatch = zc->appliedParams.cParams.minMatch;
    gpuParams.strategy = (unsigned) zc->appliedParams.cParams.strategy;


    /* Create GPU match finder - delegate to .cu implementation */
    return ZSTD_createGpuMatchFinder((ZSTD_GpuMatchFinder**) &zc->gpuMF, &gpuParams, srcSize);
}

/**
 * Check if GPU match finding should be used for this context.
 */
int ZSTD_shouldUseGpu(ZSTD_CCtx const* zc)
{
    if (!zc) return 0;
    if (!ZSTD_cudaAvailable()) return 0;
    if (!zc->gpuMF) return 0;
    return 1;
}

/* ======================================================================== */
/* ZSTD_buildSeqStore GPU Integration                                        */
/* ======================================================================== */

/**
 * GPU-accelerated version of ZSTD_buildSeqStore.
 *
 * This function replaces the CPU match finding with GPU match finding.
 * It delegates to ZSTD_compressBlock_gpu (from .cu file) which:
 * 1. Copies the input data to GPU memory
 * 2. Runs the GPU match finder kernel
 * 3. Converts GPU results to ZSTD_Sequence format
 *
 * Then this function converts ZSTD_Sequence to SeqDef format for zstd.
 */
size_t ZSTD_buildSeqStore_gpu(ZSTD_CCtx* zc, void const* src, const size_t srcSize)
{
    auto gpuMF = (ZSTD_GpuMatchFinder*) zc->gpuMF;
    unsigned rep[ZSTD_REP_NUM];

    if (!zc || !src) return ERROR(GPU_invalidParameter);
    if (srcSize < 4)
        /* Block too small for compression */
        return ZSTDbss_noCompress;

    /* Initialize GPU match finder if not already done */
    if (!gpuMF)
    {
        size_t const err = ZSTD_initGpuMatchFinderForCtx(zc);
        FORWARD_IF_ERROR(err, "GPU match finder init failed");
        gpuMF = (ZSTD_GpuMatchFinder*) zc->gpuMF;
    }

    /* Get sequence store capacity */
    const size_t maxNbSeq = zc->seqStore.maxNbSeq;

    /* Allocate sequence buffer */
    ZSTD_Sequence* seqs = malloc(maxNbSeq * sizeof(ZSTD_Sequence));
    if (!seqs) return ERROR(GPU_allocationFailed);

    /* Initialize rep codes to zero */
    rep[0] = 1;
    rep[1] = 2;
    rep[2] = 3;

    /* Run GPU match finding - delegates to .cu implementation */
    const size_t nbSeq = ZSTD_compressBlock_gpu(gpuMF, seqs, maxNbSeq, src, srcSize, rep, rep);
    if (ZSTD_isError(nbSeq))
    {
        free(seqs);
        return nbSeq;
    }

    /* Convert ZSTD_Sequence to SeqDef format and write to sequence store */
    ZSTD_resetSeqStore(&zc->seqStore);

    /* Handle the no-match case: store all data as literals */
    if (nbSeq == 0)
    {
        /* No matches found - all data is literals.
         * The entire src block is trailing literals, so the anchor is src itself. */
        ZSTD_storeLastLiterals_gpu(&zc->seqStore, (const BYTE*) src, srcSize);
        free(seqs);
        return ZSTDbss_compress;
    }

    for (size_t i = 0; i < nbSeq; i++)
    {
        SeqDef* const seq = zc->seqStore.sequences + i;
        seq->litLength = (U16) seqs[i].litLength;
        seq->mlBase = (U16) (seqs[i].matchLength - MINMATCH);
        /* GPU stores raw offset (distance); zstd format requires offBase = offset + ZSTD_REP_NUM */
        seq->offBase = seqs[i].offset + ZSTD_REP_NUM;
    }

    /* Update sequence store pointers.
     * IMPORTANT: zc->seqStore.lit must point to zc->seqStore.litStart
     * (the seqStore's own literal buffer), NOT to the source data.
     * This is because ZSTD_storeLastLiterals() will copy trailing literals
     * from the source into seqStore->lit, then advance it. */
    zc->seqStore.sequences = zc->seqStore.sequencesStart + nbSeq;
    zc->seqStore.lit = zc->seqStore.litStart;

    free(seqs);
    return ZSTDbss_compress;
}
