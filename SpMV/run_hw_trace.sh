#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHM_RESULT_BASE="${SHM_RESULT_BASE:-/dev/shm/$(id -un)}"
RESULT_ROOT="${RESULT_ROOT:-$SHM_RESULT_BASE/spmv_hw_trace_$(date +%Y%m%d_%H%M%S)}"
LATEST_RESULT_POINTER="${LATEST_RESULT_POINTER:-$SHM_RESULT_BASE/latest_spmv_trace_result_path.txt}"
NUMA_NODE="${NUMA_NODE:-0}"
N_WARMUP="${N_WARMUP:-5}"
N_REPS="${N_REPS:-30}"
TRANSPORT_KEY_MIN_SAMPLES="${TRANSPORT_KEY_MIN_SAMPLES:-20}"
TRANSPORT_KEY_MIN_TRACES="${TRANSPORT_KEY_MIN_TRACES:-20}"
TRANSPORT_KEY_SPREAD_THRESHOLD_PCT="${TRANSPORT_KEY_SPREAD_THRESHOLD_PCT:-25}"
TRANSPORT_KEY_CV_THRESHOLD_PCT="${TRANSPORT_KEY_CV_THRESHOLD_PCT:-25}"
CREATE_ARCHIVE="${CREATE_ARCHIVE:-0}"
DPU_RANK_TOPOLOGY_TSV="${DPU_RANK_TOPOLOGY_TSV:-}"
DPUS_LIST="${DPUS_LIST:-256 512}"
TASKLETS_LIST="${TASKLETS_LIST:-1 2 4 8 16}"
EXPECTED_MATRIX_SHA256="75441a848e025d78840fe638dda5c66a3597032b1bee297bd06712b7695074d8"

if [[ -z "$DPU_RANK_TOPOLOGY_TSV" || ! -r "$DPU_RANK_TOPOLOGY_TSV" ]]; then
    echo "ERROR: set DPU_RANK_TOPOLOGY_TSV to a readable dpu_rank_topology.tsv" >&2
    exit 1
fi

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
numactl --show > "$RESULT_ROOT/numactl_show.txt" 2>&1 || true
git -C "$WORKSPACE_DIR/prim-benchmarks" rev-parse HEAD > "$RESULT_ROOT/prim_git_commit.txt"
git -C "$WORKSPACE_DIR/prim-benchmarks" status --short > "$RESULT_ROOT/prim_git_status.txt"
sha256sum data/bcsstk30.mtx > "$RESULT_ROOT/matrix.sha256"
sha256sum Makefile host/app.c host/mram-management.h host/host_trace.c \
    host/host_trace.h dpu/task.c support/common.h support/matrix.h \
    support/params.h support/timer.h support/utils.h transport_key.py \
    analyze_transport_keys.py validate_hw_trace.py \
    > "$RESULT_ROOT/source.sha256"
dpu-upmem-dpurte-clang --version > "$RESULT_ROOT/dpu_compiler_version.txt" 2>&1 || true
dpu-pkg-config --cflags --libs dpu > "$RESULT_ROOT/dpu_sdk_flags.txt" 2>&1 || true
cp "$DPU_RANK_TOPOLOGY_TSV" "$RESULT_ROOT/dpu_rank_topology.tsv"
export SPMV_TRACE_DPU_RANK_TOPOLOGY_TSV="$RESULT_ROOT/dpu_rank_topology.tsv"

run_spmv() {
    local verbosity="$1"
    if command -v numactl >/dev/null 2>&1; then
        numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE" \
            ./bin/host_code -v "$verbosity" -f data/bcsstk30.mtx
    else
        ./bin/host_code -v "$verbosity" -f data/bcsstk30.mtx
    fi
}

if command -v numactl >/dev/null 2>&1; then
    TRACE_HOST_NUMA_NODE="$NUMA_NODE"
else
    TRACE_HOST_NUMA_NODE="unbound"
fi

require_correct_result() {
    local log_path="$1"
    if grep -q "Mismatch at index" "$log_path"; then
        echo "ERROR: SpMV verification mismatch in $log_path" >&2
        exit 1
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
        printf 'NR_DPUS=%s\nNR_TASKLETS=%s\nNUMA_NODE=%s\nTRACE_HOST_NUMA_NODE=%s\nN_WARMUP=%s\nN_REPS=%s\nTRANSPORT_KEY_MIN_SAMPLES=%s\nTRANSPORT_KEY_MIN_TRACES=%s\nTRANSPORT_KEY_SPREAD_THRESHOLD_PCT=%s\nTRANSPORT_KEY_CV_THRESHOLD_PCT=%s\nCREATE_ARCHIVE=%s\n' \
            "$nr_dpus" "$tasklets" "$NUMA_NODE" "$TRACE_HOST_NUMA_NODE" \
            "$N_WARMUP" "$N_REPS" "$TRANSPORT_KEY_MIN_SAMPLES" \
            "$TRANSPORT_KEY_MIN_TRACES" \
            "$TRANSPORT_KEY_SPREAD_THRESHOLD_PCT" \
            "$TRANSPORT_KEY_CV_THRESHOLD_PCT" "$CREATE_ARCHIVE" \
            > "$result_dir/config.txt"

        unset SPMV_TRACE_CSV SPMV_TRACE_RUN_ID SPMV_TRACE_REPEAT_ID \
            SPMV_TRACE_HOST_NUMA_NODE SPMV_TRACE_PROCESS_STATE \
            SPMV_TRACE_PREWARM_RUNS || true
        echo "==> Warming up $config ($N_WARMUP runs, tracing disabled)"
        for rep in $(seq 1 "$N_WARMUP"); do
            verbosity=0
            if [[ "$rep" -eq 1 ]]; then
                verbosity=1
            fi
            warmup_log="$result_dir/warmup_$(printf '%02d' "$rep").log"
            run_spmv "$verbosity" > "$warmup_log" 2>&1
            require_correct_result "$warmup_log"
        done

        echo "==> Tracing $config ($N_REPS runs)"
        for rep in $(seq 1 "$N_REPS"); do
            rep_id="$(printf '%02d' "$rep")"
            export SPMV_TRACE_CSV="$result_dir/trace_${rep_id}.csv"
            export SPMV_TRACE_RUN_ID="$config"
            export SPMV_TRACE_REPEAT_ID="$rep"
            export SPMV_TRACE_HOST_NUMA_NODE="$TRACE_HOST_NUMA_NODE"
            export SPMV_TRACE_PROCESS_STATE="fresh_process"
            export SPMV_TRACE_PREWARM_RUNS="$N_WARMUP"
            run_log="$result_dir/run_${rep_id}.log"
            run_spmv 0 > "$run_log" 2>&1
            require_correct_result "$run_log"
        done
        unset SPMV_TRACE_CSV SPMV_TRACE_RUN_ID SPMV_TRACE_REPEAT_ID \
            SPMV_TRACE_HOST_NUMA_NODE SPMV_TRACE_PROCESS_STATE \
            SPMV_TRACE_PREWARM_RUNS || true

        python3 "$SCRIPT_DIR/validate_hw_trace.py" \
            "$result_dir"/trace_[0-9][0-9].csv \
            > "$result_dir/validation.log"
        python3 "$SCRIPT_DIR/analyze_transport_keys.py" \
            --min-samples "$TRANSPORT_KEY_MIN_SAMPLES" \
            --min-traces "$TRANSPORT_KEY_MIN_TRACES" \
            --spread-threshold-pct "$TRANSPORT_KEY_SPREAD_THRESHOLD_PCT" \
            --cv-threshold-pct "$TRANSPORT_KEY_CV_THRESHOLD_PCT" \
            --output "$result_dir/transport_key_summary.csv" \
            "$result_dir"/trace_[0-9][0-9].csv \
            > "$result_dir/transport_key_analysis.log"
        echo "==> PASS $config"
    done
done

mapfile -t all_traces < <(find "$RESULT_ROOT" -mindepth 2 -maxdepth 2 \
    -type f -name 'trace_[0-9][0-9].csv' | sort)
python3 "$SCRIPT_DIR/analyze_transport_keys.py" \
    --min-samples "$TRANSPORT_KEY_MIN_SAMPLES" \
    --min-traces "$TRANSPORT_KEY_MIN_TRACES" \
    --spread-threshold-pct "$TRANSPORT_KEY_SPREAD_THRESHOLD_PCT" \
    --cv-threshold-pct "$TRANSPORT_KEY_CV_THRESHOLD_PCT" \
    --output "$RESULT_ROOT/transport_key_summary_all.csv" \
    "${all_traces[@]}" \
    > "$RESULT_ROOT/transport_key_analysis_all.log"

archive=""
if [[ "$CREATE_ARCHIVE" == "1" ]]; then
    archive="${RESULT_ROOT}.tar.gz"
    tar -czf "$archive" -C "$(dirname "$RESULT_ROOT")" \
        "$(basename "$RESULT_ROOT")"
fi
mkdir -p "$(dirname "$LATEST_RESULT_POINTER")"
printf '%s\n' "$RESULT_ROOT" | tee "$LATEST_RESULT_POINTER"
echo "Trace results: $RESULT_ROOT"
if [[ -n "$archive" ]]; then
    echo "Archive:       $archive"
else
    echo "Archive:       disabled"
fi
