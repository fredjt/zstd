/*
 * GPU radix sort + CSR row-boundary unit test (G6).
 *
 * Compiles the CUDA match-finder translation unit directly and exercises the
 * fill -> 4-pass LSD radix -> slot -> row-boundary pipeline against a host
 * reference, for several block sizes and hashLog values (14/17/22).
 *
 * Build (from the repository root, requires nvcc):
 *   nvcc -O2 -arch=sm_86 -Ilib -Ilib/common -Ilib/compress -DZSTD_CUDA \
 *        -o gpu_sort_test tests/gpu_sort_test.cu -Ilib/compress
 *   ./gpu_sort_test   # prints one line per case; exit 0 = all pass
 */
#include "zstd_cuda_matchfinder.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>

static unsigned test_hash(const unsigned char* p, unsigned hashLog)
{
    unsigned h;
    memcpy(&h, p, 4);
    return (h * 2654435761U) >> (32 - hashLog);
}

static int run_case(unsigned n, unsigned hashLog)
{
    unsigned n2 = (n + 255) & ~255u;
    unsigned char* win = (unsigned char*) malloc(n);
    unsigned long long* h_key = (unsigned long long*) malloc(n2 * 8);
    unsigned long long* h_sorted = (unsigned long long*) malloc(n2 * 8);
    unsigned* h_slot = (unsigned*) malloc(n2 * 4);
    unsigned* h_rowStart = (unsigned*) malloc((1u << hashLog) * 4);
    unsigned* h_rowEnd = (unsigned*) malloc((1u << hashLog) * 4);
    unsigned long long* d_keyA, *d_keyB; unsigned* d_bh, *d_ds, *d_slot, *d_rs, *d_re;
    cudaMalloc(&d_keyA, n2 * 8);
    cudaMalloc(&d_keyB, n2 * 8);
    cudaMalloc(&d_bh, n2 * 4); cudaMalloc(&d_ds, 256 * 4);
    cudaMalloc(&d_slot, n2 * 4);
    cudaMalloc(&d_rs, (1u << hashLog) * 4);
    cudaMalloc(&d_re, (1u << hashLog) * 4);

    for (unsigned i = 0; i < n; i++) win[i] = (unsigned char) (rand() % 4);
    for (unsigned i = 0; i < n; i++)
        h_key[i] = ((unsigned long long) test_hash(win + i, hashLog) << 17) | i;
    for (unsigned i = n; i < n2; i++) h_key[i] = ~0ULL;
    cudaMemset(d_rs, 0, (1u << hashLog) * 4);
    cudaMemset(d_re, 0, (1u << hashLog) * 4);

    cudaMemcpy(d_keyA, h_key, n2 * 8, cudaMemcpyHostToDevice);
    {
        unsigned long long* a = d_keyA;
        unsigned long long* b = d_keyB;
        unsigned const blocks = n2 / 256;
        for (unsigned pass = 0; pass < 4; pass++)
        {
            unsigned const shift = 16 + pass * 8;
            cudaMemset(d_bh, 0, (size_t) blocks * 256 * 4);
            ZSTD_gpu_radixHist<<<blocks, 256>>>(a, d_bh, shift, n2);
            ZSTD_gpu_radixScan<<<1, 256>>>(d_bh, d_ds, blocks);
            ZSTD_gpu_radixScatter<<<blocks, 256>>>(a, b, d_bh, d_ds, shift, n2);
            unsigned long long* t = a; a = b; b = t;
        }
        /* 4 passes (even) -> result in d_keyA */
    }
    ZSTD_gpu_recordSlot<<<n2 / 256, 256>>>(d_keyA, d_slot, n2);
    ZSTD_gpu_rowBounds<<<n2 / 256, 256>>>(d_keyA, d_rs, d_re, hashLog, n2);

    cudaMemcpy(h_sorted, d_keyA, n2 * 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_slot, d_slot, n2 * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rowStart, d_rs, (1u << hashLog) * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rowEnd, d_re, (1u << hashLog) * 4, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    int fails = 0;
    for (unsigned i = 1; i < n2; i++)
        if (h_sorted[i - 1] > h_sorted[i])
        {
            if (fails < 3) printf("FAIL n=%u: not sorted at %u (%llu > %llu)\n", n, i, h_sorted[i-1], h_sorted[i]);
            fails++;
        }
    /* slot inverse check */
    for (unsigned i = 0; i < n; i++)
    {
        unsigned const p = (unsigned)(h_key[i] & 0x1FFFFu);
        if (h_sorted[h_slot[p]] != h_key[i])
        {
            if (fails < 6) printf("FAIL n=%u: slot[%u]=%u but sorted[slot]=%llu key=%llu\n", n, p, h_slot[p], h_sorted[h_slot[p]], h_key[i]);
            fails++;
        }
    }
    /* rowStart/rowEnd vs host computation */
    unsigned* hs = (unsigned*) calloc(1u << hashLog, 4);
    unsigned* he = (unsigned*) calloc(1u << hashLog, 4);
    unsigned* present = (unsigned*) calloc(1u << hashLog, 4);
    int curH = -1;
    for (unsigned i = 0; i < n2; i++)
    {
        unsigned long long const key = h_sorted[i];
        if (key == ~0ULL) { if (curH >= 0) { he[curH] = i - 1; curH = -1; } continue; }
        unsigned const h = (unsigned)(key >> 17);
        if (h != curH)
        {
            if (curH >= 0) he[curH] = i - 1;
            hs[h] = i;
            present[h] = 1;
            curH = h;
        }
    }
    if (curH >= 0) he[curH] = n2 - 1; /* last row ends at the array tail */
    for (unsigned h = 0; h < (1u << hashLog); h++)
        if (present[h])
        {
            if (h_rowStart[h] != hs[h] || h_rowEnd[h] != he[h])
            {
                if (fails < 9) printf("FAIL n=%u: row h=%u dev=[%u,%u] host=[%u,%u]\n", n, h, h_rowStart[h], h_rowEnd[h], hs[h], he[h]);
                fails++;
            }
        }
    if (fails) printf("FAIL n=%u hashLog=%u: %d failures\n", n, hashLog, fails);
    else printf("PASS n=%u hashLog=%u\n", n, hashLog);
    free(win); free(h_key); free(h_sorted); free(h_slot); free(h_rowStart); free(h_rowEnd); free(hs); free(he); free(present);
    cudaFree(d_keyA); cudaFree(d_keyB); cudaFree(d_bh); cudaFree(d_slot); cudaFree(d_rs); cudaFree(d_re);
    return fails;
}

int main(void)
{
    srand(12345);
    int fails = 0;
    fails += run_case(1, 14);
    fails += run_case(4095, 14);
    fails += run_case(131069, 14);
    fails += run_case(131069, 17);
    fails += run_case(131069, 22);
    fails += run_case(512, 4);
    return fails ? 1 : 0;
}
