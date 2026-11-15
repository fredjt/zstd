# GPU match finder (POC) : resource management, fallback, and scope

Status: G8 (POC release scope). The GPU path is opt-in via `--gpu`
(`ZSTD_c_gpuMatchFinder`) and only takes effect for the `ZSTD_fast`
strategy (the level-1 class); every other strategy runs on the CPU.

## One GPU context per process

At most **one** `ZSTD_GpuMatchFinder` exists at a time. The first
compression context that asks for `--gpu` takes the device buffers;
any context created while another one still owns them (for example a
multi-thread worker context, see below) falls back to the CPU match
finder with a one-time stderr warning. The cap is a reference count in
`zstd_cuda_matchfinder.cu` (`g_gpuMFCount`) : it drops when the owning
context is freed, so a later context can take the GPU over again. Rationale: each context owns the full device buffer set below,
and N worker threads must not multiply it.

## Device memory budget (per GPU match finder)

With 128 KB blocks (`ZSTD_BLOCKSIZE_MAX`) and level-1 parameters
(`hashLog = 14`, `chainLog = 13`):

| Buffer                | Size                                  | Notes |
| --------------------- | ------------------------------------ | ----- |
| `d_window`            | srcSize + 128 KB = 256 KB            | sliding window, H2D per block |
| `d_keyA` / `d_keyB`   | 2 x 1 MB = 2 MB                      | composite (hash, position) sort keys, ping-pong |
| `d_slot`              | 512 KB                               | inverse permutation (position -> sorted index) |
| `d_rowStart` / `d_rowEnd` | 2 x 4 x 2^hashLog = 512 KB     | CSR row bounds, one row per hash |
| `d_cand`              | (128 KB + 1) x 8 B = 1 MB            | per-position (offset, matchLength) results |
| `d_blockHist`         | 512 KB                               | radix per-block histogram (512 x 256 x 4 B) |
| `d_digitStart`        | 1 KB                                 | radix per-digit output starts |
| pinned `cand_host`    | 1 MB (page-locked host)              | D2H target for `d_cand` |
| **total**             | **~5.75 MB device + 1 MB pinned**    |

The buffers scale with the block size (linear in `ZSTD_GPU_BLOCKSIZE_MAX`)
and, for the two CSR tables, with `2^hashLog`. The hashLog comes from the
applied level parameters; for the fast strategy class it stays small
(14-17). If any `cudaMalloc` fails, the whole allocation is rolled back
and the context falls back to the CPU match finder (see below).

## Fallback contract

`--gpu` is never a hard error:

1. **No CUDA device** (no driver, `CUDA_VISIBLE_DEVICES=` empty, ...):
   one-time warning on stderr, compression continues on the CPU match
   finder, output is valid and identical in structure.
2. **GPU match finder fails to initialize** (device memory allocation
   failure, process-wide context cap reached): same one-time warning,
   CPU path. The per-block dispatch checks `zc->gpuMF != NULL`, so a
   context that never got a GPU match finder simply never touches CUDA.
3. **Per-block GPU failure** (e.g. `cudaMemcpy` error mid-run): the block
   returns the corresponding `ZSTD_error_GPU_*` code. (POC scope : the
   caller is expected to treat it as a compression failure of that frame;
   transparent per-block CPU fallback is a post-POC item.)

## Multi-threading scope

`--gpu` + `--multi` is **not supported** in the POC. The zstd CLI creates
one compression context per worker thread; with the one-at-a-time cap,
only the context that wins the cap check runs on the GPU while the rest
silently run on the CPU, mixing per-frame behavior. The recommended POC
usage is single-threaded (`--single-thread` or the `ZSTD_compress*` API
without workers). Serializing all blocks through the shared GPU context
is a post-POC design item.

## Leak checking (G8 soak)

- Device side : `compute-sanitizer --tool memcheck` over a 1 MB run
  (2 blocks), a 128 KB run and a 16 MB run (156 blocks) of `zstd --gpu
  -1` : **0 errors** in all runs. The full roundtrip suite
  (`tests/gpu_tests.sh`, 30+ cases incl. the CPU-fallback case) passes.
- Host side : valgrind is not available in the build environment, so the
  create/compress/free cycle was verified with a targeted probe instead :
  after `ZSTD_freeCCtx` the device memory is released and the
  process-wide cap drops (a second context re-acquires the GPU and
  produces the GPU csize, not the CPU one). `ZSTD_freeCCtx` now owns the
  GPU match finder (the CLI FIO layer NULLs `cctx->gpuMF` after freeing
  it, so no double free).

See `tests/perf/` for the G7 benchmark harness and results.
