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
 * with zstd's internal compression flow. It handles:
 * - GPU context management (creation, initialization, destruction)
 * - Integration with ZSTD_buildSeqStore for GPU-accelerated compression
 */

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
/* G20 host-side profiling (ZSTD_GPU_HOSTPROF=1)                            */
/* ======================================================================== */
/*
 * The device profile (ZSTD_GPU_PROFILE) measures the GPU phases. This
 * measures the host-side phases of the same per-block path, to answer where
 * the host time goes : is the host simply waiting on the GPU pipeline
 * (confirming the device compute floor, G6), or is there host compute
 * (the seqstore build) worth removing ? Phases (seconds, printed per 128 KB
 * block when the match finder is freed) :
 *   0 total    : whole ZSTD_buildSeqStore_gpu (from just before the
 *                match-finding call to the end)
 *   1 gpuCall  : the match-finding call (chunk or sync) - host time spent
 *                inside it, i.e. waiting on the GPU pipeline
 *   2 seqStore : seqstore reset + sequence conversion + literal copy
 */
#include <stdio.h>
#include <time.h>
static double gpuHostNowW(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static double   g_hp[3];
static unsigned g_hpBlocks;
static int      g_hpOn = -1;
static int gpuHostProfOn(void)
{
    if (g_hpOn < 0) g_hpOn = (getenv("ZSTD_GPU_HOSTPROF") != NULL);
    return g_hpOn;
}
/* Accumulate the seqStore (t1 -> now) and total (t0 -> now) phases. */
static void gpuHostProfEnd(double t0, double t1)
{
    if (!gpuHostProfOn()) return;
    {
        double const t2 = gpuHostNowW();
        g_hp[2] += t2 - t1;
        g_hp[0] += t2 - t0;
        g_hpBlocks++;
    }
}
void ZSTD_gpuHostProfReport(void)
{
    if (!gpuHostProfOn() || g_hpBlocks == 0) return;
    {
        static const char* const names[3] = { "total", "gpuCall", "seqStore" };
        double const total = g_hp[0];
        fprintf(stderr, "[ZSTD_GPU_HOSTPROF] per 128KB block (avg over %u blocks)\n", g_hpBlocks);
        for (int p = 0; p < 3; p++) {
            double const avg = 1000.0 * g_hp[p] / (double)g_hpBlocks;
            double const pct = (total > 0) ? 100.0 * g_hp[p] / total : 0.0;
            fprintf(stderr, "  %-11s %9.4f ms  %5.1f%%\n", names[p], avg, pct);
        }
    }
}

/* ======================================================================== */
/* GPU Match Finder Context Management                                       */
/* ======================================================================== */

/**
 * Initialize the GPU match finder from CCtx parameters.
 * Called when GPU mode is first requested.
 */
/* POC scope (G8) : at most one GPU match finder per process - enforced by
 * the reference count in ZSTD_createGpuMatchFinder (.cu), which fails here
 * with an allocation error when another context already owns the GPU. The
 * caller (zstd_compress.c) turns that into a one-time warning plus the CPU
 * match finder. */
size_t ZSTD_initGpuMatchFinderForCtx(ZSTD_CCtx* zc)
{
    ZSTD_GpuCompressionParameters gpuParams = {0};

    if (!zc) return ERROR(GPU_invalidParameter);

    /* Configure GPU parameters from CCtx params */
    gpuParams.hashLog   = zc->appliedParams.cParams.hashLog;
    gpuParams.chainLog  = zc->appliedParams.cParams.chainLog;
    gpuParams.searchLog = zc->appliedParams.cParams.searchLog;
    gpuParams.minMatch  = zc->appliedParams.cParams.minMatch;
    gpuParams.strategy  = (unsigned) zc->appliedParams.cParams.strategy;
    gpuParams.windowLog = zc->appliedParams.cParams.windowLog;

    /* Create GPU match finder - delegate to .cu implementation */
    {
        size_t const err = ZSTD_createGpuMatchFinder((ZSTD_GpuMatchFinder**) &zc->gpuMF, &gpuParams, ZSTD_BLOCKSIZE_MAX);
        FORWARD_IF_ERROR(err, "GPU match finder creation failed");
    }
    return 0;
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
 * Runs the GPU match finder (a single-block batch) for the block, then
 * converts the returned (position, offset, matchLength) sequences into the
 * SeqStore format (literal runs + sequences) exactly like the CPU
 * matchfinders do.
 */
size_t ZSTD_buildSeqStore_gpu(ZSTD_CCtx* zc, void const* src, const size_t srcSize)
{
    ZSTD_GpuMatchFinder* gpuMF = (ZSTD_GpuMatchFinder*) zc->gpuMF;
    unsigned rep[ZSTD_REP_NUM];
    size_t i;

    if (!zc || !src) return ERROR(GPU_invalidParameter);

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
    ZSTD_GpuSequence* seqs = (ZSTD_GpuSequence*)malloc(maxNbSeq * sizeof(ZSTD_GpuSequence));
    if (!seqs) return ERROR(GPU_allocationFailed);

    /* The GPU always emits raw offsets (rep = 0) : repcodes are not used,
     * so the rep arrays are left at zero. */
    rep[0] = rep[1] = rep[2] = 0;

    /* Run GPU match finding - delegates to .cu implementation. The chunk
     * pipeline (G19) is driven by the frame chunk loop, which stages a
     * look-ahead of blocks so the GPU runs ahead of the host encoder; it
     * advances nextWait per block and must therefore be called for every
     * block (even tiny ones). Otherwise fall back to the synchronous
     * single-block path. */
    double const ht0 = gpuHostProfOn() ? gpuHostNowW() : 0;   /* G20 */
    size_t nbSeq;
    if (gpuMF->chunkMode)
    {
        nbSeq = ZSTD_gpuChunk_processBlock(gpuMF, (ZSTD_GpuSequence*)seqs, maxNbSeq);
    }
    else
    {
        if (srcSize < 4)
        {
            /* Block too small for compression */
            free(seqs);
            return ZSTDbss_noCompress;
        }
        nbSeq = ZSTD_compressBlock_gpu(gpuMF, (ZSTD_Sequence*)seqs, maxNbSeq, src, srcSize, rep, rep);
    }
    if (ZSTD_isError(nbSeq))
    {
        free(seqs);
        return nbSeq;
    }
    double const ht1 = gpuHostProfOn() ? gpuHostNowW() : 0;   /* G20 */
    if (gpuHostProfOn()) g_hp[1] += ht1 - ht0;

    /* Convert (position, offset, matchLength) to SeqDef format and write to
     * the sequence store. Each sequence owns the literals before its match
     * (litLength = position - previousEnd), and the block's tail is one last
     * literal run. */
    ZSTD_resetSeqStore(&zc->seqStore);

    /* Handle the no-match case: store all data as literals. */
    if (nbSeq == 0)
    {
        ZSTD_storeLastLiterals_gpu(&zc->seqStore, (const BYTE*) src, srcSize);
        gpuHostProfEnd(ht0, ht1);
        free(seqs);
        return ZSTDbss_compress;
    }

    {
        size_t prevEnd = 0;
        const BYTE* const srcB = (const BYTE*) src;
        for (i = 0; i < nbSeq; i++)
        {
            SeqDef* const seq = zc->seqStore.sequences + i;
            size_t const pos = seqs[i].position;
            size_t const litLength = pos - prevEnd;

            /* Long-length tracking, exactly like the CPU ZSTD_storeSeqOnly() :
             * a match longer than 65535 + MINMATCH (or a literal run longer
             * than 65535) does not fit in the U16 field. The encoder recovers
             * the full length from longLengthPos/longLengthType. A block is at
             * most 128 KB and matches are non-overlapping, so at most one long
             * length exists. */
            if (litLength > 0xFFFF)
            {
                assert(zc->seqStore.longLengthType == ZSTD_llt_none);
                zc->seqStore.longLengthType = ZSTD_llt_literalLength;
                zc->seqStore.longLengthPos = (U32) i;
            }
            seq->litLength = (U16) litLength;
            {
                size_t const mlBase = seqs[i].matchLength - MINMATCH;
                if (mlBase > 0xFFFF)
                {
                    assert(zc->seqStore.longLengthType == ZSTD_llt_none);
                    zc->seqStore.longLengthType = ZSTD_llt_matchLength;
                    zc->seqStore.longLengthPos = (U32) i;
                }
                seq->mlBase = (U16) mlBase;
            }
            /* GPU stores raw offset (distance); zstd format requires
             * offBase = offset + ZSTD_REP_NUM. */
            seq->offBase = seqs[i].offset + ZSTD_REP_NUM;

            /* Copy the literal run before this match. */
            ZSTD_memcpy(zc->seqStore.lit, srcB + prevEnd, litLength);
            zc->seqStore.lit += litLength;
            prevEnd = pos + seqs[i].matchLength;
        }
        /* Trailing literals. */
        {
            size_t const lastLLSize = srcSize - prevEnd;
            ZSTD_storeLastLiterals_gpu(&zc->seqStore, srcB + prevEnd, lastLLSize);
        }
    }

    /* zc->seqStore.lit now points just past the last copied literal, and
     * sequences points just past the last sequence. */
    zc->seqStore.sequences = zc->seqStore.sequencesStart + nbSeq;
    gpuHostProfEnd(ht0, ht1);

    free(seqs);
    return ZSTDbss_compress;
}
