/*
 * G7 : CPU vs --gpu compression benchmark harness.
 *
 * Usage : gpu_cpu_bench <mode:cpu|gpu> <level> <sizeMB> <dataset:rnd|rep>
 * Prints one line : mode level MB dataset csize MB/s
 *
 * Build (repository root) :
 *   gcc -O2 -Ilib -Ilib/common -Ilib/compress tests/perf/gpu_cpu_bench.c \
 *       -o tests/perf/gpu_cpu_bench lib/libzstd.a \
 *       <objdir>/zstd_cuda_matchfinder.o \
 *       -lpthread -lcudart -L/usr/local/cuda/lib64 -lstdc++ -lgcc_s
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <zstd.h>
#include <stdint.h>

/* This tree does not expose ZSTD_isError in the public header. */
#define IS_ERR(x) (((int64_t) (x)) < 0)

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec / 1e9;
}

int main(int argc, char** argv)
{
    if (argc < 5) return 2;
    int const gpu = !strcmp(argv[1], "gpu");
    int const level = atoi(argv[2]);
    size_t const n = (size_t) atoi(argv[3]) * 1024 * 1024;
    int const rnd = !strcmp(argv[4], "rnd");

    char* src = (char*) malloc(n);
    char* dst = (char*) malloc(ZSTD_compressBound(n));
    if (!src || !dst) return 3;

    if (rnd)
    {
        unsigned s = 12345;
        for (size_t i = 0; i < n; i++) { s = s * 1103515245u + 12345u; src[i] = (char)(s >> 16); }
    }
    else
    {
        static const char pat[] = "the quick brown fox jumps over the lazy dog. 0123456789. ";
        size_t const pl = sizeof(pat) - 1;
        for (size_t i = 0; i < n; i++) src[i] = pat[i % pl];
    }

    ZSTD_CCtx* c = ZSTD_createCCtx();
    ZSTD_CCtx_setParameter(c, ZSTD_c_compressionLevel, level);
    if (gpu) ZSTD_CCtx_setParameter(c, ZSTD_c_gpuMatchFinder, 1);

    /* warm-up (first GPU block includes context setup) */
    {
        ZSTD_CCtx* w = ZSTD_createCCtx();
        ZSTD_CCtx_setParameter(w, ZSTD_c_compressionLevel, level);
        if (gpu) ZSTD_CCtx_setParameter(w, ZSTD_c_gpuMatchFinder, 1);
        ZSTD_compress2(w, dst, ZSTD_compressBound(n), src, n);
        ZSTD_freeCCtx(w);
    }

    double const t0 = now_sec();
    size_t const r = ZSTD_compress2(c, dst, ZSTD_compressBound(n), src, n);
    double const dt = now_sec() - t0;
    ZSTD_freeCCtx(c);

    if (IS_ERR(r))
    {
        printf("%s L%d %zuMB %s ERROR %zu\n", gpu ? "gpu" : "cpu", level, n >> 20, rnd ? "rnd" : "rep", r);
        return 1;
    }
    printf("%s L%d %zuMB %s %zu %.1f\n", gpu ? "gpu" : "cpu", level, n >> 20, rnd ? "rnd" : "rep", r, (double) n / dt / 1e6);
    return 0;
}