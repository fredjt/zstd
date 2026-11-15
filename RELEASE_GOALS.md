# RELEASE_GOALS.md — GPU-Accelerated zstd: Working POC

Purpose of this fork: a GPU-accelerated version of zstd. This document defines the
goals for the **working proof of concept (POC)**. Every item was investigated and
verified against the code and CI on **2026-11-15** (fork `dev` at `522ffaa49`,
upstream `dev` merged in; GPU skeleton in commit `10902bcdd`).

---

## Definition of the POC ("done" means)

On a machine with an NVIDIA GPU and the CUDA toolkit:

1. The default (non-CUDA) build is **unchanged and fully green** — all CI jobs pass,
   upstream test suite passes, zero GPU code compiled in.
2. A CUDA-enabled build compiles, links, and produces a `zstd` binary where
   `zstd --gpu <file>`:
   - uses the GPU for the **match-finding phase** of compression,
   - produces a **valid zstd frame** that `zstd -d` (and an independent upstream
     binary) decodes back to the **byte-identical** original, for a corpus covering
     edge cases (see G5),
   - is **at least as fast as the CPU** for the match-finding phase on
     block-sized (128 KB) inputs at a comparable cLevel, with compression ratio
     within ~5% of CPU greedy (numbers to be measured in G7 and documented).
3. Graceful behavior without a GPU: clear warning + CPU fallback, never a crash.
4. Documented: how to build, how to use, what the POC does and does not do.

---

## Current state (verified findings)

### CI is red on every job
- `facebook/zstd/commit` (push): `short-tests-0`, `short-tests-1`, `regression-test`
  all fail.
  - With `CFLAGS="-Werror ..."`: `lib/common/error_private.c:22` fails with
    `-Werror=switch-enum` — 4 of the 14 GPU error codes added to `zstd_errors.h`
    have no case in `ERR_getErrorString()`:
    `ZSTD_error_GPU_matchFinder_notInitialized` (121), `ZSTD_error_GPU_cudaMemcpyFailed`
    (122), `ZSTD_error_GPU_kernelLaunchFailed` (123), `ZSTD_error_GPU_blockTooLarge`
    (124).
  - Without `-Werror` (nightly regression): the lib compiles (warnings only), but
    linking the `zstd` program fails:
    `undefined reference to ZSTD_createGpuMatchFinder / ZSTD_cudaAvailable /
    ZSTD_compressBlock_gpu` (see below).
- `facebook/zstd/nightly`: failing daily since the GPU commit landed (first observed
  2026-11-01). Same root causes.

### The program link failure (why a *plain* `make` build breaks)
`programs/Makefile` does **not** link `libzstd.a` — it compiles the library sources
*directly* into the binary:
`ZSTDLIB_COMPRESS_SRC := $(sort $(ZSTD_COMPRESS_FILES))`, and `ZSTD_COMPRESS_FILES`
is a `wildcard` over `lib/compress/*.c` (`lib/libzstd.mk:153`). That automatically
picks up `zstd_cuda_matchfinder_wrapper.c`, which references the `.cu`-defined
symbols. The `.cu` is only compiled when the make variable `ZSTD_CUDA=1` **and**
nvcc exist — so every plain build of the program fails at link.
(Reproduced locally with a clean checkout in a space-free path.)

### The CUDA build path is wired up in both build systems but non-functional
- **make** (`ZSTD_CUDA=1`, nvcc present):
  1. `ZSTD_GPU_MATCH_OBJ` is appended to `ZSTD_STATICLIB_OBJ` via `+=` in
     `lib/libzstd.mk:267`, but `lib/Makefile:126` runs **later** with `:=`
     (`ZSTD_STATICLIB_OBJ := $(addprefix ...)`), clobbering the append. Verified:
     with a stub nvcc, `ar t libzstd.a` contains only the wrapper object, the `.cu`
     object is missing.
  2. Nothing defines the `ZSTD_CUDA` **preprocessor macro** in make builds, so the
     integration code in `zstd_compress.c` (`#ifdef ZSTD_CUDA`, lines 33/2278/3255)
     and `programs/fileio.c` (51/1502/1520) is compiled out. Even with the object
     in the archive, the GPU path would be dead code.
  3. The `.cu` is built with `-arch=sm_50` (`lib/libzstd.mk:231`) — pre-2014
     Maxwell; it will not run on current hardware (e.g. RTX 30xx, sm_86).
- **CMake** (`-DZSTD_BUILD_GPU=ON`):
  1. The `.cu` is built as a separate shared lib `zstd_gpu_shared`; the wrapper is
     added to the main library sources. `ZSTD_CUDA` is a compile definition on
     `zstd_gpu_shared` only (`build/cmake/lib/CMakeLists.txt:87`); for the static
     target it propagates (line 218 `PUBLIC`), for the shared target it is
     `PRIVATE` (line 193) — so the shared build's main library still lacks the
     integration code.
  2. `CMAKE_CUDA_ARCHITECTURES "80;89;90"` omits sm_86 (the local GPUs); sm_80 is
     forward-compatible, but 86 should be added.
  3. The wrapper `zstd_cuda_matchfinder_wrapper.c` is compiled **as C** but contains
     C++ syntax (`constexpr` at line 55, `auto` at line 97). It happens to compile
     on C23-default compilers (local gcc 16; nightly gcc 15.2, with warnings), but
     is a hard error on C17 toolchains — including the CI jobs pinned to gcc-7/gcc-8
     in `dev-short-tests.yml`.

### GPU implementation: not yet correct, and not parallel
Code review of `lib/compress/zstd_cuda_matchfinder.cu` / `_wrapper.c` / `.h`:

1. **Missing stream synchronization (data race).** In `ZSTD_runGpuMatchFinding` the
   comment starting at `.cu:458` (`/* Synchronize stream to ensure kernel is
   complete before copying results`) was never closed and swallows the
   `cudaStreamSynchronize` call. The kernel is launched async on a custom stream and
   the results are read back with a synchronous `cudaMemcpy` on the default stream —
   no ordering between them. Non-deterministic corrupted `nbSeq`/sequences; almost
   certainly the source of the earlier segfaults.
2. **Literals are never stored.** In `ZSTD_buildSeqStore_gpu` (wrapper), the
   `nbSeq > 0` path fills `SeqDef` entries but **never copies any literal bytes**
   into `zc->seqStore.litStart`; `seqStore.lit` is reset to the (untouched) buffer
   start. The compressed frame's literal pool would contain stale/garbage bytes for
   any block containing both matches and literals (i.e., most blocks).
   (The `nbSeq == 0` path does copy all-literals correctly.)
3. **Window constraint not enforced from the CCtx.** The kernel's `maxDistance`
   (`windowLog`) is derived from `srcSize` only (clamped 15..27, `.cu:~352`), not
   from `zc->appliedParams.cParams.windowLog/windowSize`. Offsets can exceed the
   window advertised in the frame header → decoder rejects the block.
4. **Vestigial, incorrect repcode handling.** The wrapper passes the same array as
   both `rep` and `prevRep` and seeds `{1,2,3}`; `ZSTD_convertResultsToSequences`
   updates them as `rep[1] = rep[0]; rep[0] = lastOffset` (shift is wrong). Harmless
   only because the GPU never emits repcodes (`rep = 0`, raw offsets), but it is
   misleading and should be removed or implemented properly.
5. **`ZSTD_estimateGpuMatchFinderMemory` arithmetic is wrong**: after
   `total += (1 << hashLog)` it does `total *= sizeof(unsigned)`, multiplying the
   window bytes too.
6. **Per-block overhead is enormous.** For every 128 KB block the wrapper rebuilds
   the *entire* hash+chain tables on the host (malloc, memset, sequential loop) and
   H2D-copies the whole `2^hashLog` and `2^chainLog` word arrays. Only the ~128 K
   touched entries change per block.
7. **The kernel is sequential.** `ZSTD_gpu_hashChainMatchFinder` does
   `if (threadIdx.x != 0 || blockIdx.x != 0) return;` — one thread does everything,
   byte-by-byte, in global memory. There is no parallelism; with launch + copy
   overhead the GPU path is expected to be **slower** than the CPU. This is the
   core gap the POC must close (G6).
8. **One GPU context per `ZSTD_CCtx`** (`ZSTD_resetCCtx_internal`,
   `zstd_compress.c:~2277`): in multithreaded mode every worker thread's context
   gets its own device tables (2^hashLog·4 B + 2^chainLog·4 B each).

### Local environment notes
- This checkout's path contains a space ("Unsloth **Studio**"), which zstd's
  Makefiles cannot handle: `LIB_SRCDIR = $(dir $(realpath $(lastword $(MAKEFILE_LIST))))`
  (`lib/libzstd.mk:83`) splits the path, the source wildcard yields nothing, and
  `make` silently builds an **empty** `libzstd.a`. **Use CMake for all local builds
  in this checkout** (verified: configures, builds `libzstd` + `zstd`, roundtrip OK).
  (Upstream could later harden this; out of scope for the POC.)
- Local machine: RTX 3070 + RTX 3060 (both sm_86), driver present, **no CUDA
  toolkit installed** — install CUDA 12.x (provides nvcc + headers) before G3.
- Green baseline established: non-GPU CMake build, `zstd` CLI roundtrip verified on
  incompressible data and on compressible data at `-19`.

---

## Goals (in execution order)

### G1. Restore CI to green (unblock everything)
1. Add the 4 missing cases to the `ERR_getErrorString()` switch in
   `lib/common/error_private.c` (with sensible strings, e.g. "GPU match finder:
   not initialized", "GPU match finder: cudaMemcpy failed", "GPU match finder:
   kernel launch failed", "GPU match finder: block too large").
2. Keep the error-code set in `zstd_errors.h` and the switch exhaustive; consider
   a unit test or `static_assert`-style check so they can never drift again.
- **Accept:** `facebook/zstd/commit` fully green on a push; locally
  `make -C lib CFLAGS="-O0 -Werror -pedantic"` (in a space-free checkout) is clean.

### G2. Default (non-CUDA) build stays pristine
1. Make the source list explicitly **exclude** `zstd_cuda_matchfinder_wrapper.c`
   when CUDA is off (fix the wildcard pickup in `lib/libzstd.mk`, and correspond
   `ZSTDLIB_COMPRESS_SRC` in `programs/Makefile`). This is what fixes the plain
   `make -C programs zstd` link failure.
2. Ensure no GPU symbol, warning, or behavior change leaks into non-CUDA builds
   (the `ZSTD_c_gpuMatchFinder` parameter must remain a documented no-op there).
3. Run the upstream test suite in the non-CUDA build.
- **Accept:** all CI jobs green on a plain push; `tests/playTests.sh` passes;
  `zstd` binary from a non-CUDA build is behavior-identical to upstream dev.

### G3. Make the CUDA build actually build, link, and select the GPU path
1. **Wrapper as portable C:** rewrite `zstd_cuda_matchfinder_wrapper.c` in plain C11
   (no `constexpr`/`auto`/other C++-isms) so it compiles on the gcc-7/gcc-8 CI jobs
   and on any C17 toolchain. (Alternatively rename to `.cpp` and build as C++ —
   C11 is the better fit for zstd's codebase.)
2. **make path:** add the wrapper to `ZSTD_LOCAL_SRC` only when CUDA is enabled;
   fix the `+=`/`:=` ordering so `ZSTD_GPU_MATCH_OBJ` survives into
   `ZSTD_STATICLIB_OBJ` (move the append in `lib/Makefile` *after* the `:=`, or use
   `+=`); define the `ZSTD_CUDA` preprocessor macro (`CPPFLAGS += -DZSTD_CUDA`) in
   both lib and programs when enabled; put the `.cu` object in the OBJDIR conf dir
   instead of the source tree; update `-arch=sm_50` to current archs (e.g.
   `sm_52,sm_80,sm_86,sm_90` or `native`).
3. **CMake path:** give the main zstd targets (static and shared) the `ZSTD_CUDA`
   definition when `ZSTD_BUILD_GPU=ON`; add sm_86 to `CMAKE_CUDA_ARCHITECTURES`;
   verify the `zstd_gpu_shared` linkage is actually used by the `zstd` program.
4. Install the CUDA toolkit locally (nvcc) and build both paths end-to-end on this
   machine.
- **Accept:** in a space-free checkout, `make ZSTD_CUDA=1` and
  `cmake -DZSTD_BUILD_GPU=ON` each produce a `zstd` binary in which the GPU code
  paths are compiled in and linked (verify with `nm`/`strings`, and a runtime log
  line when `--gpu` is selected).

### G4. Make the GPU path produce valid frames (correctness)
Fix, in order of severity:
1. **Restore the stream sync** in `ZSTD_runGpuMatchFinding`: close the broken
   comment (`.cu:458`) and keep `cudaStreamSynchronize` (or a synchronous launch)
   before the D→H results copy.
2. **Store the literals:** in `ZSTD_buildSeqStore_gpu`, walk the sequences and copy
   every literal run `[prevEnd, seqPos)` into `zc->seqStore.lit` (mirroring what the
   CPU matchfinder does, including the trailing literals after the last match), and
   set `nbSeq`/pointers consistently. The all-literals (`nbSeq == 0`) path must keep
   working.
3. **Enforce the window:** pass the CCtx's `windowSize`
   (`zc->appliedParams.cParams.windowSize`) into the kernel and skip candidates with
   `pos - matchIndex > windowSize` (replacing the srcSize-derived `maxDistance`).
4. **Repcode hygiene:** either remove the fake repcode plumbing (GPU always emits
   raw offsets, `rep = 0`) or implement repcode emission properly; fix the shift in
   `ZSTD_convertResultsToSequences` if kept.
5. Fix `ZSTD_estimateGpuMatchFinderMemory` (sum the four terms correctly).
6. Verify with `compute-sanitizer` (racecheck/memcheck) that no device race or OOB
   remains.
- **Accept:** `zstd --gpu` output for the G5 corpus roundtrips byte-identical
  through `zstd -d` and through an independent upstream-built binary; `zstd -t`
  passes on every output; sanitizer runs clean.

### G5. GPU correctness test suite
1. Add a GPU test script (e.g. `tests/gpu_tests.sh`, invoked from
   `tests/playTests.sh` when a GPU + CUDA build are present, skipped gracefully
   otherwise) covering:
   - sizes: 1 B, 3 B (MINMATCH-1), 4 B, exactly `ZSTD_BLOCKSIZE_MAX` (128 KB),
     128 KB ± 1, 1 MB, 10 MB, 100 MB (multi-block), stdin streaming vs file;
   - content: all-same-byte (nbSeq==0 blocks), random (incompressible), repeated
     sentence (many short matches), periodic (long matches), dictionary-like;
   - cLevels: 1, 3, 9, 19, and `--gpu` vs `--no-gpu` output both valid (ratio may
     differ); `--gpu` on a CPU-only machine must warn and still produce valid output
     (CPU fallback).
2. For each case: compress (`--gpu`) → `zstd -t` → decompress → `cmp` with original.
- **Accept:** suite passes on this machine (3070/3060); suite is a no-skip-failure
  on non-GPU CI (skipped with a notice).

### G6. Make match finding actually parallel on the GPU (the point of the POC)
The current single-thread kernel cannot beat the CPU. POC design (document
  tradeoffs in a short design note):
1. Keep tables **on device across blocks** (allocate once, update incrementally
   per block — only the ~128 K touched entries, not whole-table H2D copies).
2. Parallelize the search: one thread per candidate position walks a bounded chain
   (depth `2^searchLog`) and counts matches (4-byte compare); emit per-position
   best candidates. Hash/chain **insertion** is inherently sequential; acceptable
   POC options: (a) sequential per-block table build on device (128 K steps —
   fine), (b) parallel first-wins bucket build + chain stitching, (c) host-side
   build of the *delta* only. Pick the simplest that meets G7's speed target.
3. Resolve overlapping candidates into non-overlapping sequences (sequential host
   pass over GPU output, or a small device prefix pass) — must reproduce
   "greedy, non-overlapping, longest-in-chain-order" semantics of the CPU
   `ZSTD_fast`-class strategy.
4. Decide the POC scope: greedy/fast strategy only first (zstd levels 1-3 class);
   document that lazy/bt strategies stay on CPU.
- **Accept:** on a 128 KB block at a comparable level, GPU match finding is
  ≥ 1.0× CPU (target ≥ 1.5×) **and** the full `--gpu` end-to-end speed is ≥ CPU
  for ≥ 1 MB inputs; ratio within ~5% of CPU greedy. (Measurements in G7.)

### G7. Performance harness and published numbers
1. Add a benchmark script (fullbench + a small shell harness): CPU vs `--gpu`
   across cLevels {1, 3, 9, 19} and sizes {1 MB, 16 MB, 256 MB, 1 GB}, reporting
   end-to-end MiB/s **and** the match-finding phase time (timing instrumentation
   or `ZSTD_compressBegin/End` deltas).
2. Record results (GPU model, driver, CUDA version) in `doc/` or README.
- **Accept:** a results table exists in the repo; at least one configuration shows
  GPU ≥ CPU end-to-end; if not, a written analysis of the bottleneck.

### G8. Resource management, robustness, fallback semantics
1. **One shared GPU context** per process (or at least per compression session),
   not per `ZSTD_CCtx`/MT thread; document the device-memory budget
   (window + 2^hashLog·4 B + 2^chainLog·4 B + 786 KB pinned results) and cap
   hashLog/chainLog to what fits (`cudaMalloc` failure → clean fallback, not crash).
2. **Fallback contract:** `--gpu` with no GPU/CUDA → warning on stderr + CPU path,
   identical output validity; never a hard error.
3. **MT scope for the POC:** document that `--gpu` + `--multi` is unsupported
   initially (or serialize GPU blocks); guard against N× device allocation.
4. No leaks: verify with `valgrind` (host) + `compute-sanitizer` (device) over a
   soak of G5.
- **Accept:** soak test passes; memory behavior documented; fallback verified on a
  CPU-only path (e.g. `CUDA_VISIBLE_DEVICES=`).

### G9. Documentation and POC release
1. README section: prerequisites (driver + CUDA toolkit), build instructions for
   CMake and make, usage (`zstd --gpu`), parameter semantics (`ZSTD_c_gpuMatchFinder`),
   POC limitations (G6 scope, no streaming/dictionary/LZDM on GPU, MT restrictions),
   and the G7 results table.
2. Release the POC: tag `gpu-poc-0.1` on `dev` (or a dedicated branch + PR to the
   fork's default), with a release note summarizing goals met (G1–G8) and known
   limitations.
- **Accept:** a fresh machine with driver + toolkit follows the README to a working
  `--gpu` binary in < 15 minutes.

---

## Explicit non-goals for the POC
- GPU acceleration of literals/entropy coding or decompression (match finding only).
- Streaming API (`ZSTD_compressStream*`), dictionary compression, LZDM/long-range
  mode, patch-from on GPU.
- Repcode optimization and non-greedy strategies on GPU.
- Non-CUDA backends (HIP/ROCm/SYCL), Windows, macOS.
- Upstream-friendly API design (public GPU handles are internal for now);
  `ZSTD_c_gpuMatchFinder` stays experimental.

## Immediate next steps (critical path)
1. **G1** — 4-line fix to `error_private.c`; restores all CI.
2. **G2** — exclude the wrapper from non-CUDA source lists; fixes plain `make` builds
   (nightly + anyone without nvcc).
3. **G3** — build wiring (C11 rewrite, `-DZSTD_CUDA`, archs, link); install nvcc.
4. **G4** — sync + literals + window fixes; then G5 validation.
5. **G6** — parallel kernel; then G7/G8 measurements, G9 release.
