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

---

## Refreshed numbers (post G10-G17)

Measured with the CMake CUDA build, single-threaded `-T1`, level 1, on the
same RTX 3070. The G10-G14 pipeline (double-buffered H2D, batched launches,
8-byte prefilter, on-device greedy) and the G15 greedy-strategy enablement
are in. `ZSTD_GPU_PROFILE=1` gives the per-phase table below.

### Phase breakdown (per 128 KB block, ms, level 1)

| dataset | size  | H2D    | fillKeys | radix  | search | greedy | D2H    | total  |
| ------- | ----- | ------ | -------- | ------ | ------ | ------ | ------ | ------ |
| rnd     | 1 MB  | 0.023  | 0.086    | 0.108  | 0.022  | 0.009  | 0.011  | 0.258  |
| rep     | 1 MB  | 0.023  | 0.090    | 0.106  | 0.094  | 2.112  | 0.009  | 2.433  |
| rnd     | 1 GB  | 0.022  | 0.011    | 0.089  | 0.016  | 0.004  | 0.010  | 0.153  |
| rep     | 1 GB  | 0.022  | 0.012    | 0.082  | 0.079  | 1.911  | 0.007  | 2.114  |

- **Random data :** `radix` (the 4-pass fingerprint sort) and `fillKeys`
  dominate; the greedy is a no-op (no candidates). ~0.15 ms/block at 1 GB.
- **Periodic data :** the on-device `greedy` dominates (~90 %) because a
  single thread extends one very long match (the whole block) sequentially.
  This is the main remaining speed gap.

### End-to-end MiB/s (level 1, `-T1`)

| MB    | rnd : cpu | rnd : gpu | rep : cpu | rep : gpu |
| ----- | --------- | --------- | --------- | --------- |
| 16    | 533       | 28        | 533       | 21        |
| 1024  | 648       | 255       | 1014      | 52        |

### Ratio (level 1, csize)

| input    | cpu  | gpu  | note |
| -------- | ---- | ---- | ---- |
| 1 MB rep | 168  | 161  | gpu within ~4 % |
| 1 GB rep | 97.9 KB | 117.6 KB | gpu ~20 % larger (per-block overhead paid 8192x) |

### Status vs acceptance

- **G5 correctness :** full suite passes, including 10 MB / 100 MB inputs in
  default (multi-threaded) mode, after the G16 per-job reset.
- **G14 ratio :** the 5.8x baseline penalty (573 KB vs 98 KB) is gone; the GPU
  now matches the CPU within ~4 % on 1 MB and is ~20 % larger on 1 GB (the
  per-block literal/table overhead paid 8192 times). Not yet within the ~5 %
  1 GB target.
- **G15 greedy strategy :** levels 3-4 now run the device path (verified
  active) with ratio parity (167 B = 167 B at levels 3-4 on 1 MB).
- **G16 MT :** `--multi --gpu` produces valid frames and matches the CPU ratio
  within 0.3 % (100 MB rep : 9652 B vs 9623 B).
- **G6 speed clause (GPU >= CPU for match finding) :** not yet met - the GPU is
  2.5-25x slower end-to-end. The on-device greedy extension of long matches
  (periodic) and the per-block launch/sync overhead (random) are the gaps to
  close.

---

## Refreshed numbers (post G18)

G18 attacked the two G6 bottlenecks: the single-thread greedy extension
(G18.1, now a warp-cooperative first-mismatch scan) and the per-candidate
global atomic in the search (G18.4, now a per-CUDA-block reduction). G18.2/
G18.5 removed the dead `rowBounds`/`d_blockCounter` work and made the
profiler measure all 8 phases. Same build, `-T1`, level 1, RTX 3070.

### Phase breakdown (per 128 KB block, ms, level 1)

| dataset | size  | H2D    | fillKeys | radix  | recordSlot | search | insert | greedy | D2H    | total  |
| ------- | ----- | ------ | -------- | ------ | ---------- | ------ | ------ | ------ | ------ | ------ |
| rnd     | 1 GB  | 0.019  | 0.006    | 0.071  | 0.010      | 0.015  | 0.007  | 0.003  | 0.005  | 0.135  |
| rep     | 1 GB  | 0.019  | 0.005    | 0.070  | 0.010      | 0.009  | 0.010  | 0.124  | 0.005  | 0.252  |

- **Periodic data :** the `greedy` is now 0.124 ms (was 1.911, 15x) - the
  warp-cooperative extension of the one whole-block match. `radix` (0.070) is
  now the second-largest phase.
- **Random data :** `radix` (0.071, the 2-pass fingerprint sort) dominates;
  the greedy is a no-op. ~0.135 ms/block.

### End-to-end MiB/s (level 1, `-T1`)

| MB    | rnd : gpu (G17) | rnd : gpu (G18) | rep : gpu (G17) | rep : gpu (G18) |
| ----- | --------------- | --------------- | --------------- | --------------- |
| 1024  | 255             | 492             | 52              | 364             |

### Status vs acceptance (post G18)

- **G6 speed clause (GPU >= CPU) :** still not met. rep 364 MiB/s vs CPU
  ~1014; rnd 492 MiB/s vs CPU ~648. The per-block GPU compute (0.252 ms rep /
  0.135 ms rnd) is algorithmically above the CPU's hash-table cost (no sort),
  so it is the floor; the host also submits one block per GPU round-trip (no
  overlap). See the G6 analysis in `RELEASE_GOALS.md`. Follow-ups: host-side
  batching/pipelining (~1.3-1.5x) and a sort-free device search (the only path
  to a competitive per-block GPU time).
- **Correctness :** output byte-identical to the G17 baseline on both 1 GB
  datasets; G5 suite passes.

## G19 : host-side chunk pipeline (2026-11-20, RTX 3070)

The frame chunk loop now stages a look-ahead of blocks so the GPU batch
pipeline runs ahead of the host encoder (2 batches in flight; `ZSTD_GPU_SYNC=1`
forces the old synchronous path). Level 1, one-shot, end-to-end MiB/s:

| MB    | rnd : sync | rnd : chunk | rep : sync | rep : chunk |
| ----- | ---------- | ----------- | ---------- | ----------- |
| 1     | 43.8       | 45.3        | 45.8       | 47.2        |
| 16    | 282.6      | 385.1       | 221.3      | 399.7       |
| 256   | 483.5      | 1197.2      | 377.7      | 1140.8      |

1.4-1.8x at 16 MB, 2.5-3.0x at 256 MB vs the synchronous path. The gain grows
with size: with 16 blocks per batch the GPU amortizes its ~10 kernel launches
and the H2D latency of the next batch is hidden behind the current batch's
compute. At 1 MB (8 blocks, half a batch) there is little to overlap, hence
the small gain.

Same comparison on the RTX 5070 Ti (sm_120, 32-core host), level 1:

| MB    | rnd : sync | rnd : chunk | rep : sync | rep : chunk | rep : cpu |
| ----- | ---------- | ----------- | ---------- | ----------- | --------- |
| 16    | 644.3      | 1083.0      | 415.1      | 859.9       | 15920.4   |
| 256   | 888.6      | 1859.6      | 510.6      | 1521.6      | 14723.8   |

1.7-3.0x vs the synchronous path. The G6 speed clause (GPU >= CPU end-to-end)
is not met on either machine - the per-block GPU compute is the floor (see
the G6 analysis in `RELEASE_GOALS.md`); the remaining follow-up is a
sort-free device search.

Correctness: G5 suite passes; 1 GB rnd/rep roundtrip byte-identical; ratio
unchanged (1 GB rep level 1: GPU 97934 B vs CPU 97853 B, +0.08%; level 3:
byte-identical csize).

