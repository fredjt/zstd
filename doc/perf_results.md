# G7 : CPU vs --gpu performance

Harness : `tests/perf/run_perf.sh` (drives `tests/perf/gpu_cpu_bench.c`),
wall time of a full `ZSTD_compress2` (one warm-up run excluded), end-to-end
MiB/s of the input size. Datasets : `rnd` = PRNG bytes, `rep` = 57-byte
periodic text. Raw results : `tests/perf/results_*.txt`.

## Environment

- GPU : NVIDIA GeForce RTX 3070 (sm_86)
- Driver : 620.70 ; CUDA toolkit : 13.4 (nvcc)
- zstd : this tree, G6 revision (single-threaded `ZSTD_compress2`)

## Results

End-to-end MiB/s (input size / wall time).

| MB    | lvl | rnd : cpu | rnd : gpu | rep : cpu | rep : gpu |
| ----- | --- | --------- | --------- | --------- | --------- |
| 1     | 1   | 1713.9    | 189.6     | 4089.9    | 178.7     |
| 16    | 1   | 2175.5    | 255.2     | 6416.6    | 256.3     |
| 256   | 1   | 1791.9    | 282.3     | 7002.6    | 241.1     |
| 1024  | 1   | 1828.7    | 259.4     | 7208.3    | 241.7     |
| 1     | 3   | 903.0     | 458.6     | 1323.9    | 516.4     |
| 16    | 3   | 2212.7    | 1517.2    | 3036.9    | 3028.5    |
| 256   | 3   | 1600.7    | 1445.9    | 3357.9    | 3327.0    |
| 1024  | 3   | 1553.9    | 1571.7    | 3432.9    | 3351.7    |
| 1     | 9   | 246.0     | 138.7     | 185.8     | 146.2     |
| 16    | 9   | 778.7     | 713.2     | 932.1     | 819.5     |
| 256   | 9   | 965.8     | 959.1     | 1010.4    | 1001.5    |
| 1024  | 9   | 1042.1    | 958.9     | 983.4     | 1000.4    |
| 1     | 19  | 11.2      | 12.0      | 131.7     | 113.7     |
| 16    | 19  | 7.2       | 7.2       | 306.6     | 321.5     |
| 256   | 19  | 6.2       | 5.6       | 739.1     | 728.8     |
| 1024  | 19  | 6.2       | 6.2       | 758.6     | 762.0     |

Note : the GPU match finder only runs on the `fast` strategy (level 1 class).
Levels 3/9/19 use the same CPU strategy in both modes, so their numbers are
the same code path modulo the (tiny) per-block GPU-context overhead.

## Analysis

1. **Levels 3/9/19 (CPU strategy in both modes) : parity.** End-to-end
   speeds are within a few percent of each other at every size (e.g. 1 GB
   level 9 rep : cpu 983.4 vs gpu 1000.4 MiB/s). These satisfy the
   "at least one configuration shows GPU >= CPU end-to-end" acceptance
   clause, but only because both modes run the CPU match finder there.
2. **Level 1 (the only level where the GPU actually matches) : the GPU is
   ~7x slower on random data and ~25-30x slower on the periodic file**
   (e.g. 1 GB : cpu 1828.7 vs gpu 259.4 MiB/s). Per 128 KB block the GPU
   path costs ~450-500 us (H2D, fill, 4 radix passes, per-position search
   with a capped count loop, D2H, host greedy, one stream sync) while the
   CPU fast strategy spends ~40-80 us. The per-block
   `cudaStreamSynchronize` and ~10 kernel launches per block dominate the
   fixed overhead (8192 blocks per 1 GB).
3. **Ratio : the GPU matches the CPU when the blocks are equal.** 128 KB
   block : 80 B = 80 B ; 1 MB block : 164 B = 164 B (bit-identical sizes).
   On the multi-MB periodic file the GPU csize looks worse (e.g. 1 GB rep :
   cpu 98 KB vs gpu 573 KB) because the CPU context cuts larger blocks and
   therefore pays the per-block literal/table overhead fewer times. That is
   a block-partitioning effect, not a match-quality effect.
4. **Bottlenecks to close (follow-ups) :**
   - pipeline H2D of block N+1 with the host greedy of block N (one stream
     currently serializes everything);
   - batch blocks per launch (fewer launches/syncs, one sync per batch);
   - pre-filter candidates with an 8-byte check before the per-position
     count loop on device;
   - move the greedy scan onto the device (it only needs the candidates and
     the window).

