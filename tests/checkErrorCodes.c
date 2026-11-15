/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */

/* checkErrorCodes : validation tool for libzstd error strings
 *
 * Every ZSTD_ErrorCode in [ZSTD_error_no_error, ZSTD_error_maxCode) must map to
 * a dedicated, human-readable string in ERR_getErrorString() (exposed through
 * ZSTD_getErrorName() / ZSTD_getErrorString()). When a case is missing from that
 * switch, the code silently falls back to the generic "Unspecified error code"
 * string. This tool fails when that happens, so the error code set in
 * zstd_errors.h and the string table in error_private.c can never drift apart
 * again (regression guard for the GPU match finder error codes).
 */

#include <stdio.h>   /* printf */
#include <string.h>  /* strcmp */
#include "zstd.h"            /* ZSTD_ErrorCode, ZSTD_getErrorString */
#include "error_private.h"   /* ERR_getErrorString, ERR_enum */

/* Full list of the error code members declared in zstd_errors.h.
 * The enum is sparse (members carry explicit values), so the test cannot
 * walk a raw integer range; it checks every member listed here.
 * Note : the compile-time exhaustiveness is already enforced by
 * -Werror=switch-enum on ERR_getErrorString() in the library build;
 * this tool is a runtime guard that the string table keeps matching.
 */
static const ZSTD_ErrorCode kAllCodes[] = {
    ZSTD_error_no_error,
    ZSTD_error_GENERIC,
    ZSTD_error_prefix_unknown,
    ZSTD_error_version_unsupported,
    ZSTD_error_frameParameter_unsupported,
    ZSTD_error_frameParameter_windowTooLarge,
    ZSTD_error_corruption_detected,
    ZSTD_error_checksum_wrong,
    ZSTD_error_literals_headerWrong,
    ZSTD_error_dictionary_corrupted,
    ZSTD_error_dictionary_wrong,
    ZSTD_error_dictionaryCreation_failed,
    ZSTD_error_parameter_unsupported,
    ZSTD_error_parameter_combination_unsupported,
    ZSTD_error_parameter_outOfBound,
    ZSTD_error_tableLog_tooLarge,
    ZSTD_error_maxSymbolValue_tooLarge,
    ZSTD_error_maxSymbolValue_tooSmall,
    ZSTD_error_cannotProduce_uncompressedBlock,
    ZSTD_error_stabilityCondition_notRespected,
    ZSTD_error_stage_wrong,
    ZSTD_error_init_missing,
    ZSTD_error_memory_allocation,
    ZSTD_error_workSpace_tooSmall,
    ZSTD_error_dstSize_tooSmall,
    ZSTD_error_srcSize_wrong,
    ZSTD_error_dstBuffer_null,
    ZSTD_error_noForwardProgress_destFull,
    ZSTD_error_noForwardProgress_inputEmpty,
    /* not stable */
    ZSTD_error_frameIndex_tooLarge,
    ZSTD_error_seekableIO,
    ZSTD_error_dstBuffer_wrong,
    ZSTD_error_srcBuffer_wrong,
    ZSTD_error_sequenceProducer_failed,
    ZSTD_error_externalSequences_invalid,
    /* GPU match finder (not stable) */
    ZSTD_error_GPU_invalidParameter,
    ZSTD_error_GPU_cudaNotAvailable,
    ZSTD_error_GPU_invalidHashLog,
    ZSTD_error_GPU_invalidChainLog,
    ZSTD_error_GPU_invalidSearchLog,
    ZSTD_error_GPU_invalidMinMatch,
    ZSTD_error_GPU_invalidSearchDepth,
    ZSTD_error_GPU_allocationFailed,
    ZSTD_error_GPU_cudaStreamCreationFailed,
    ZSTD_error_GPU_cudaAllocationFailed,
    ZSTD_error_GPU_matchFinder_notInitialized,
    ZSTD_error_GPU_cudaMemcpyFailed,
    ZSTD_error_GPU_kernelLaunchFailed,
    ZSTD_error_GPU_blockTooLarge,
};

#define KALLCODES_N ((size_t)(sizeof(kAllCodes) / sizeof(kAllCodes[0])))

int main(void)
{
    size_t i;
    size_t missing = 0;
    /* query an out-of-range code to capture the pointer of the fallback
     * string returned by the default case of ERR_getErrorString() */
    const char* const fallback = ERR_getErrorString((ERR_enum)(0 - 1000));

    if (fallback == NULL) {
        printf("checkErrorCodes : internal error : missing fallback string \n");
        return 1;
    }

    for (i = 0; i < KALLCODES_N; i++) {
        const char* const name = ZSTD_getErrorString(kAllCodes[i]);
        if (name == NULL || name == fallback || !strcmp(name, fallback)) {
            printf("checkErrorCodes : error code %d maps to the fallback string \"%s\" \n",
                   (int)kAllCodes[i], fallback);
            missing++;
        }
    }

    if (missing) {
        printf("checkErrorCodes : FAILED : %zu error code(s) missing a dedicated string \n", missing);
        return 1;
    }

    printf("checkErrorCodes : OK : all %zu error codes have a dedicated string \n", KALLCODES_N);
    return 0;
}
