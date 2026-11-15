#!/bin/sh
# GPU match-finding correctness tests (POC goal G5)
#
# Exercises the --gpu compression path of a CUDA-enabled zstd binary and
# verifies that every compressed output
#   1. passes `zstd -t`
#   2. decompresses back to the byte-identical original
#
# The suite is skipped (exit 0) when the binary was built without CUDA or
# when no GPU is available, so it never fails on non-GPU CI.

set -u

ZSTD_BIN="${ZSTD_BIN:-../../programs/zstd}"
WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT INT TERM

fail=0
note()    { echo "  [gpu] $*"; }
failcase(){ echo "  [gpu] FAIL: $*"; fail=1; }

# gen <name> <bytes> <type> : write $WORKDIR/<name> with content of <type>
gen() {
    name=$1; bytes=$2; type=$3
    case $type in
        same)      head -c "$bytes" /dev/zero > "$WORKDIR/$name" ;;
        random)    head -c "$bytes" /dev/urandom > "$WORKDIR/$name" ;;
        repeated)  yes "the quick brown fox jumps over the lazy dog." | head -c "$bytes" > "$WORKDIR/$name" ;;
        periodic)
            head -c 4096 /dev/urandom > "$WORKDIR/.pat"
            i=0
            while [ "$i" -lt "$bytes" ]; do
                cat "$WORKDIR/.pat" >> "$WORKDIR/$name"
                i=$((i + 4096))
            done
            head -c "$bytes" "$WORKDIR/$name" > "$WORKDIR/$name.tmp" && mv "$WORKDIR/$name.tmp" "$WORKDIR/$name"
            ;;
        dictlike)  cat ../lib/compress/*.c 2>/dev/null | head -c "$bytes" > "$WORKDIR/$name" ;;
    esac
}

# rt <label> <level> <srcfile> [flags...]
rt() {
    label=$1; lvl=$2; src=$3; shift 3
    z="$WORKDIR/$label.zst"; out="$WORKDIR/$label.out"
    if ! $ZSTD_BIN -v1 -"$lvl" --gpu $@ -f "$src" -o "$z" 2>/dev/null; then
        failcase "$label compress (level $lvl)"; return
    fi
    if ! $ZSTD_BIN -t "$z" 2>/dev/null; then
        failcase "$label zstd -t"; return
    fi
    if ! $ZSTD_BIN -d -f "$z" -o "$out" 2>/dev/null; then
        failcase "$label decompress"; return
    fi
    if ! cmp -s "$src" "$out"; then
        failcase "$label roundtrip mismatch"; return
    fi
    note "$label (level $lvl) OK"
}

# stdin variant: same as rt but feed the source through stdin
rt_stdin() {
    label=$1; lvl=$2; src=$3
    z="$WORKDIR/$label.zst"; out="$WORKDIR/$label.out"
    if ! $ZSTD_BIN -v1 -"$lvl" --gpu -f < "$src" -o "$z" 2>/dev/null; then
        failcase "$label stdin compress"; return
    fi
    if ! $ZSTD_BIN -t "$z" 2>/dev/null; then
        failcase "$label stdin zstd -t"; return
    fi
    if ! $ZSTD_BIN -d -f "$z" -o "$out" 2>/dev/null; then
        failcase "$label stdin decompress"; return
    fi
    if ! cmp -s "$src" "$out"; then
        failcase "$label stdin roundtrip mismatch"; return
    fi
    note "$label stdin (level $lvl) OK"
}

# --- Detection : skip gracefully when no CUDA binary or no GPU -------------
if [ ! -x "$ZSTD_BIN" ]; then
    echo "  [gpu] no zstd binary at $ZSTD_BIN - skipping GPU tests"
    exit 0
fi
if ! $ZSTD_BIN --help 2>/dev/null | grep -q -- "--gpu"; then
    echo "  [gpu] binary built without CUDA support - skipping GPU tests"
    exit 0
fi
gen probe 65536 repeated
if ! $ZSTD_BIN -v3 -3 --gpu -f "$WORKDIR/probe" -o "$WORKDIR/probe.zst" 2>&1 \
        | grep -q "Using GPU-accelerated match finder"; then
    echo "  [gpu] no GPU available (or CUDA runtime missing) - skipping GPU tests"
    exit 0
fi
note "GPU detected - running correctness suite"

# --- Sizes : edge cases around MINMATCH and the 128 KB block boundary ------
for s in 1 3 4 131071 131072 131073 1048576 10485760 104857600; do
    gen sz_$s $s repeated
    rt "sz_$s" 3 "$WORKDIR/sz_$s"
done

# --- Contents at 1 MB -------------------------------------------------------
for t in same random repeated periodic dictlike; do
    gen ct_$t 1048576 $t
    rt "ct_$t" 3 "$WORKDIR/ct_$t"
done

# --- cLevels on match-rich inputs ------------------------------------------
for l in 1 3 9 19; do
    rt "lvl_${l}_blk"  $l "$WORKDIR/sz_131072"
    rt "lvl_${l}_1mb"  $l "$WORKDIR/ct_periodic"
done

# --- stdin vs file ----------------------------------------------------------
rt_stdin "stdin_blk" 3 "$WORKDIR/sz_131072"
rt_stdin "stdin_1mb" 9 "$WORKDIR/ct_repeated"

# --- Window enforcement (G4.3) ----------------------------------------------
# Two identical 128 KB halves : matches sit exactly at distance 128 KB.
gen win 131072 random
cat "$WORKDIR/win" "$WORKDIR/win" > "$WORKDIR/win_dup"
rt "win_default" 3 "$WORKDIR/win_dup"   # level 3 window = 128 KB : exact boundary

# --- CPU-only fallback : warn, but still produce valid output ---------------
gen fb 1048576 repeated
if CUDA_VISIBLE_DEVICES= $ZSTD_BIN -v3 -3 --gpu -f "$WORKDIR/fb" -o "$WORKDIR/fb.zst" 2>/dev/null; then
    if $ZSTD_BIN -t "$WORKDIR/fb.zst" 2>/dev/null \
       && $ZSTD_BIN -d -f "$WORKDIR/fb.zst" -o "$WORKDIR/fb.out" 2>/dev/null \
       && cmp -s "$WORKDIR/fb" "$WORKDIR/fb.out"; then
        note "CPU fallback (CUDA_VISIBLE_DEVICES=) OK"
    else
        failcase "CPU fallback roundtrip"
    fi
else
    failcase "CPU fallback compress"
fi

# --- Baseline : plain (no --gpu) runs in a CUDA build stay valid ------------
for t in repeated random; do
    if $ZSTD_BIN -v1 -3 -f "$WORKDIR/ct_$t" -o "$WORKDIR/base_$t.zst" 2>/dev/null \
       && $ZSTD_BIN -t "$WORKDIR/base_$t.zst" 2>/dev/null \
       && $ZSTD_BIN -d -f "$WORKDIR/base_$t.zst" -o "$WORKDIR/base_$t.out" 2>/dev/null \
       && cmp -s "$WORKDIR/ct_$t" "$WORKDIR/base_$t.out"; then
        note "baseline (no --gpu) $t OK"
    else
        failcase "baseline (no --gpu) $t"
    fi
done

if [ "$fail" -ne 0 ]; then
    echo "  [gpu] GPU test suite FAILED"
    exit 1
fi
echo "  [gpu] GPU test suite passed"
exit 0
