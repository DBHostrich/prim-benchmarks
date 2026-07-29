#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-$WORKSPACE_DIR/red_hw_trace_$(date +%Y%m%d_%H%M%S)}"
NUMA_NODE="${NUMA_NODE:-0}"
N_WARMUP="${N_WARMUP:-5}"
N_REPS="${N_REPS:-30}"
DPUS_LIST="${DPUS_LIST:-256 512}"
TASKLETS_LIST="${TASKLETS_LIST:-1 2 4 8 16}"
INPUT_ELEMENTS=6553600

mkdir -p "$RESULT_ROOT"
cd "$SCRIPT_DIR"

uname -a > "$RESULT_ROOT/uname.txt"
lscpu > "$RESULT_ROOT/lscpu.txt"
numactl --hardware > "$RESULT_ROOT/numa.txt" 2>&1 || true
git -C "$WORKSPACE_DIR/prim-benchmarks" rev-parse HEAD \
    > "$RESULT_ROOT/prim_git_commit.txt"
git -C "$WORKSPACE_DIR/prim-benchmarks" status --short \
    > "$RESULT_ROOT/prim_git_status.txt"
sha256sum Makefile host/app.c host/host_trace.c host/host_trace.h dpu/task.c \
    support/common.h support/params.h support/timer.h \
    > "$RESULT_ROOT/source.sha256"
dpu-upmem-dpurte-clang --version \
    > "$RESULT_ROOT/dpu_compiler_version.txt" 2>&1 || true
dpu-pkg-config --cflags --libs dpu \
    > "$RESULT_ROOT/dpu_sdk_flags.txt" 2>&1 || true

run_red() {
    if command -v numactl >/dev/null 2>&1; then
        numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE" \
            ./bin/host_code -w 1 -e 3 -i "$INPUT_ELEMENTS" -x 1
    else
        ./bin/host_code -w 1 -e 3 -i "$INPUT_ELEMENTS" -x 1
    fi
}

require_correct_result() {
    local log_path="$1"
    if ! grep -q "Outputs are equal" "$log_path"; then
        echo "ERROR: RED correctness marker is absent in $log_path" >&2
        exit 1
    fi
}

for nr_dpus in $DPUS_LIST; do
    for tasklets in $TASKLETS_LIST; do
        config="RED_${nr_dpus}dpu_${tasklets}tl"
        result_dir="$RESULT_ROOT/$config"
        mkdir -p "$result_dir"

        echo "==> Building $config"
        make clean
        make NR_DPUS="$nr_dpus" NR_TASKLETS="$tasklets" BL=10 \
            VERSION=SINGLE SYNC=HAND TYPE=INT64 PERF=0 ENERGY=0 all \
            > "$result_dir/build.log" 2>&1
        sha256sum bin/host_code bin/dpu_code > "$result_dir/binaries.sha256"
        printf \
            'NR_DPUS=%s\nNR_TASKLETS=%s\nBL=10\nVERSION=SINGLE\nSYNC=HAND\nTYPE=INT64\nPERF=0\nENERGY=0\nNUMA_NODE=%s\nINPUT_ELEMENTS=%s\nN_WARMUP_IN_PROCESS=1\nN_REPS_IN_PROCESS=3\n' \
            "$nr_dpus" "$tasklets" "$NUMA_NODE" "$INPUT_ELEMENTS" \
            > "$result_dir/config.txt"

        unset RED_TRACE_CSV RED_TRACE_DPUS_CSV RED_TRACE_RUN_ID \
            RED_TRACE_REPEAT_ID || true
        echo "==> Warming up $config ($N_WARMUP processes, tracing disabled)"
        for rep in $(seq 1 "$N_WARMUP"); do
            warmup_log="$result_dir/warmup_$(printf '%02d' "$rep").log"
            run_red > "$warmup_log" 2>&1
            require_correct_result "$warmup_log"
        done

        echo "==> Tracing $config ($N_REPS processes)"
        for rep in $(seq 1 "$N_REPS"); do
            rep_id="$(printf '%02d' "$rep")"
            export RED_TRACE_CSV="$result_dir/trace_${rep_id}.csv"
            export RED_TRACE_DPUS_CSV="$result_dir/trace_${rep_id}_dpus.csv"
            export RED_TRACE_RUN_ID="$config"
            export RED_TRACE_REPEAT_ID="$rep"
            run_log="$result_dir/run_${rep_id}.log"
            run_red > "$run_log" 2>&1
            require_correct_result "$run_log"
        done
        unset RED_TRACE_CSV RED_TRACE_DPUS_CSV RED_TRACE_RUN_ID \
            RED_TRACE_REPEAT_ID || true

        python3 "$SCRIPT_DIR/validate_hw_trace.py" \
            "$result_dir"/trace_[0-9][0-9].csv \
            > "$result_dir/validation.log"
        echo "==> PASS $config"
    done
done

archive="${RESULT_ROOT}.tar.gz"
tar -czf "$archive" -C "$(dirname "$RESULT_ROOT")" \
    "$(basename "$RESULT_ROOT")"
printf '%s\n' "$RESULT_ROOT" \
    | tee "$WORKSPACE_DIR/latest_red_trace_result_path.txt"
echo "Trace results: $RESULT_ROOT"
echo "Archive:       $archive"
