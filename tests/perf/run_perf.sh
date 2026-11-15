#!/bin/sh
# G7 : CPU vs --gpu benchmark driver.
# Usage : tests/perf/run_perf.sh [sizeMB list] [dataset list]
#   defaults : sizes "1 16 256 1024", datasets "rnd rep"
# Results go to stdout and to tests/perf/results_<timestamp>.txt.
set -e

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# BUILD : the zstd build tree holding lib/libzstd.a and the compiled .cu
# object (defaults to the repository root, i.e. an in-tree build).
BUILD="${BUILD:-$ROOT}"
export PATH=/usr/local/cuda/bin:$PATH

LEVELS="1 3 9 19"
SIZES="${1:-1 16 256 1024}"
DATASETS="${2:-rnd rep}"

CUOBJ=$(find "$BUILD/programs/obj" -name "zstd_cuda_matchfinder.o" 2>/dev/null | head -1)
if [ -z "$CUOBJ" ]; then
    echo "no zstd_cuda_matchfinder.o found - build with ZSTD_CUDA=1 first" >&2
    exit 1
fi

gcc -O2 -I"$ROOT/lib" -I"$ROOT/lib/common" -I"$ROOT/lib/compress" \
    "$ROOT/tests/perf/gpu_cpu_bench.c" -o "$ROOT/tests/perf/gpu_cpu_bench" \
    "$BUILD/lib/libzstd.a" "$CUOBJ" -lpthread -lcudart -L/usr/local/cuda/lib64 -lstdc++ -lgcc_s

OUT="$ROOT/tests/perf/results_$(date +%Y%m%d_%H%M%S).txt"
{
    echo "# G7 benchmark run $(date)"
    echo "# GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    echo "# Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    echo "# CUDA: $(nvcc --version 2>/dev/null | grep -o 'release [0-9.]*' | head -1)"
    echo "# zstd: $(git -C "$ROOT" log --oneline -1 | cut -c1-12)"
    printf "%-6s %-4s %-7s %-8s %-10s %8s\n" mode lvl MB dataset csize_B MB_s
} | tee "$OUT"

for lvl in $LEVELS; do
    for size in $SIZES; do
        for ds in $DATASETS; do
            for mode in cpu gpu; do
                line=$(timeout 600 "$ROOT/tests/perf/gpu_cpu_bench" $mode $lvl $size $ds 2>/dev/null) \
                    || line="$mode L$lvl ${size}MB $ds TIMEOUT"
                echo "$line" | tee -a "$OUT"
            done
        done
    done
done

echo "results: $OUT"