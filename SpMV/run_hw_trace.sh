#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-$WORKSPACE_DIR/spmv_hw_trace_$(date +%Y%m%d_%H%M%S)}"
NUMA_NODE="${NUMA_NODE:-0}"
N_WARMUP="${N_WARMUP:-5}"
N_REPS="${N_REPS:-30}"
DPUS_LIST="${DPUS_LIST:-256 512}"
TASKLETS_LIST="${TASKLETS_LIST:-1 16}"
EXPECTED_MATRIX_SHA256="75441a848e025d78840fe638dda5c66a3597032b1bee297bd06712b7695074d8"

mkdir -p "$RESULT_ROOT"
cd "$SCRIPT_DIR"

actual_matrix_sha256="$(sha256sum data/bcsstk30.mtx | awk '{print $1}')"
if [[ "$actual_matrix_sha256" != "$EXPECTED_MATRIX_SHA256" ]]; then
    echo "ERROR: bcsstk30.mtx SHA256 is $actual_matrix_sha256" >&2
    echo "       expected $EXPECTED_MATRIX_SHA256" >&2
    exit 1
fi

uname -a > "$RESULT_ROOT/uname.txt"
lscpu > "$RESULT_ROOT/lscpu.txt"
numactl --hardware > "$RESULT_ROOT/numa.txt" 2>&1 || true
git -C "$WORKSPACE_DIR/prim-benchmarks" rev-parse HEAD > "$RESULT_ROOT/prim_git_commit.txt"
git -C "$WORKSPACE_DIR/prim-benchmarks" status --short > "$RESULT_ROOT/prim_git_status.txt"
sha256sum data/bcsstk30.mtx > "$RESULT_ROOT/matrix.sha256"
dpu-upmem-dpurte-clang --version > "$RESULT_ROOT/dpu_compiler_version.txt" 2>&1 || true
dpu-pkg-config --cflags --libs dpu > "$RESULT_ROOT/dpu_sdk_flags.txt" 2>&1 || true

run_spmv() {
    local verbosity="$1"
    if command -v numactl >/dev/null 2>&1; then
        numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE" \
            ./bin/host_code -v "$verbosity" -f data/bcsstk30.mtx
    else
        ./bin/host_code -v "$verbosity" -f data/bcsstk30.mtx
    fi
}

for nr_dpus in $DPUS_LIST; do
    for tasklets in $TASKLETS_LIST; do
        config="SpMV_${nr_dpus}dpu_${tasklets}tl"
        result_dir="$RESULT_ROOT/$config"
        mkdir -p "$result_dir"

        echo "==> Building $config"
        make clean
        make NR_DPUS="$nr_dpus" NR_TASKLETS="$tasklets" all \
            > "$result_dir/build.log" 2>&1
        sha256sum bin/host_code bin/dpu_code > "$result_dir/binaries.sha256"
        printf 'NR_DPUS=%s\nNR_TASKLETS=%s\nNUMA_NODE=%s\n' \
            "$nr_dpus" "$tasklets" "$NUMA_NODE" > "$result_dir/config.txt"

        unset SPMV_TRACE_CSV SPMV_TRACE_RUN_ID SPMV_TRACE_REPEAT_ID || true
        echo "==> Warming up $config ($N_WARMUP runs, tracing disabled)"
        for rep in $(seq 1 "$N_WARMUP"); do
            verbosity=0
            if [[ "$rep" -eq 1 ]]; then
                verbosity=1
            fi
            run_spmv "$verbosity" > "$result_dir/warmup_$(printf '%02d' "$rep").log" 2>&1
        done

        echo "==> Tracing $config ($N_REPS runs)"
        for rep in $(seq 1 "$N_REPS"); do
            rep_id="$(printf '%02d' "$rep")"
            export SPMV_TRACE_CSV="$result_dir/trace_${rep_id}.csv"
            export SPMV_TRACE_RUN_ID="$config"
            export SPMV_TRACE_REPEAT_ID="$rep"
            run_spmv 0 > "$result_dir/run_${rep_id}.log" 2>&1
        done
        unset SPMV_TRACE_CSV SPMV_TRACE_RUN_ID SPMV_TRACE_REPEAT_ID || true

        python3 "$SCRIPT_DIR/validate_hw_trace.py" "$result_dir"/trace_*.csv \
            > "$result_dir/validation.log"
        echo "==> PASS $config"
    done
done

archive="${RESULT_ROOT}.tar.gz"
tar -czf "$archive" -C "$(dirname "$RESULT_ROOT")" "$(basename "$RESULT_ROOT")"
printf '%s\n' "$RESULT_ROOT" | tee "$WORKSPACE_DIR/latest_spmv_trace_result_path.txt"
echo "Trace results: $RESULT_ROOT"
echo "Archive:       $archive"
