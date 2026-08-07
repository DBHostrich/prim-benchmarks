#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-/tmp/bdang/spmv_v8_numa0_$(date +%Y%m%d_%H%M%S)}"
LATEST_RESULT_POINTER="${LATEST_RESULT_POINTER:-/tmp/bdang/latest_spmv_v8_numa0_path.txt}"
NUMA_NODE="${NUMA_NODE:-0}"
EXCLUDED_SYSFS_RANKS="${EXCLUDED_SYSFS_RANKS:-5}"
DPUS_LIST="${DPUS_LIST:-64 128 256 512 1024 1216}"
TASKLETS_LIST="${TASKLETS_LIST:-1}"
N_WARMUP="${N_WARMUP:-3}"
N_REPS="${N_REPS:-30}"
TRANSPORT_KEY_MIN_SAMPLES="${TRANSPORT_KEY_MIN_SAMPLES:-20}"
TRANSPORT_KEY_MIN_TRACES="${TRANSPORT_KEY_MIN_TRACES:-20}"
TRANSPORT_KEY_SPREAD_THRESHOLD_PCT="${TRANSPORT_KEY_SPREAD_THRESHOLD_PCT:-25}"
TRANSPORT_KEY_CV_THRESHOLD_PCT="${TRANSPORT_KEY_CV_THRESHOLD_PCT:-25}"
CREATE_ARCHIVE="${CREATE_ARCHIVE:-0}"
DPU_RANK_TOPOLOGY_TSV="${DPU_RANK_TOPOLOGY_TSV:-}"
EXPECTED_MATRIX_SHA256="75441a848e025d78840fe638dda5c66a3597032b1bee297bd06712b7695074d8"
TRANSPORT_KEY_VERSION="v8_physical_cpu_dpu_topology"

if [[ -z "$DPU_RANK_TOPOLOGY_TSV" || ! -r "$DPU_RANK_TOPOLOGY_TSV" ]]; then
    echo "ERROR: set DPU_RANK_TOPOLOGY_TSV to a readable dpu_rank_topology.tsv" >&2
    exit 1
fi
if [[ "$NUMA_NODE" != "0" ]]; then
    echo "ERROR: this experiment is defined for NUMA_NODE=0" >&2
    exit 1
fi

mkdir -p "$RESULT_ROOT/artifacts"
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
ls -l /dev/dpu_rank* > "$RESULT_ROOT/dpu_rank_devices.txt" 2>&1 || true
git -C "$WORKSPACE_DIR/prim-benchmarks" rev-parse HEAD \
    > "$RESULT_ROOT/prim_git_commit.txt"
git -C "$WORKSPACE_DIR/prim-benchmarks" status --short \
    > "$RESULT_ROOT/prim_git_status.txt"
sha256sum data/bcsstk30.mtx > "$RESULT_ROOT/matrix.sha256"
sha256sum Makefile host/app.c host/mram-management.h host/host_trace.c \
    host/host_trace.h dpu/task.c support/common.h support/matrix.h \
    support/params.h support/timer.h support/utils.h transport_key.py \
    analyze_transport_keys.py validate_hw_trace.py select_rank_paths.py \
    summarize_scale_results.py run_interleaved_scale_trace.sh \
    > "$RESULT_ROOT/source.sha256"
dpu-upmem-dpurte-clang --version \
    > "$RESULT_ROOT/dpu_compiler_version.txt" 2>&1 || true
dpu-pkg-config --cflags --libs dpu \
    > "$RESULT_ROOT/dpu_sdk_flags.txt" 2>&1 || true
cp "$DPU_RANK_TOPOLOGY_TSV" "$RESULT_ROOT/dpu_rank_topology.tsv"
sha256sum "$RESULT_ROOT/dpu_rank_topology.tsv" \
    > "$RESULT_ROOT/dpu_rank_topology.sha256"
export SPMV_TRACE_DPU_RANK_TOPOLOGY_TSV="$RESULT_ROOT/dpu_rank_topology.tsv"

if command -v numactl >/dev/null 2>&1; then
    TRACE_HOST_NUMA_NODE="$NUMA_NODE"
else
    echo "ERROR: numactl is required for the NUMA0 experiment" >&2
    exit 1
fi

configs=()
for nr_dpus in $DPUS_LIST; do
    if (( nr_dpus % 64 != 0 )); then
        echo "ERROR: DPU count $nr_dpus is not a whole number of ranks" >&2
        exit 1
    fi
    rank_count=$((nr_dpus / 64))
    rank_paths="$(python3 "$SCRIPT_DIR/select_rank_paths.py" \
        "$SPMV_TRACE_DPU_RANK_TOPOLOGY_TSV" \
        --numa-node "$NUMA_NODE" \
        --exclude-sysfs-ranks "$EXCLUDED_SYSFS_RANKS" \
        --rank-count "$rank_count" --field rank_path)"
    sysfs_ranks="$(python3 "$SCRIPT_DIR/select_rank_paths.py" \
        "$SPMV_TRACE_DPU_RANK_TOPOLOGY_TSV" \
        --numa-node "$NUMA_NODE" \
        --exclude-sysfs-ranks "$EXCLUDED_SYSFS_RANKS" \
        --rank-count "$rank_count" --field sysfs_rank_id)"
    for tasklets in $TASKLETS_LIST; do
        configs+=("${nr_dpus}:${tasklets}:${rank_paths}:${sysfs_ranks}")
    done
done
if (( ${#configs[@]} < 2 )); then
    echo "ERROR: interleaved collection requires at least two configurations" >&2
    exit 1
fi
printf 'phase,round,slot,config,start_wall_ns,end_wall_ns\n' \
    > "$RESULT_ROOT/schedule.csv"

run_spmv() {
    local verbosity="$1"
    numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE" \
        ./bin/host_code -v "$verbosity" -f data/bcsstk30.mtx
}

require_correct_result() {
    local log_path="$1"
    if grep -q "Mismatch at index" "$log_path"; then
        echo "ERROR: SpMV verification mismatch in $log_path" >&2
        exit 1
    fi
}

config_name() {
    local spec="$1"
    local nr_dpus tasklets remainder
    nr_dpus="${spec%%:*}"
    remainder="${spec#*:}"
    tasklets="${remainder%%:*}"
    printf 'SpMV_%sdpu_%stl' "$nr_dpus" "$tasklets"
}

activate_config() {
    local spec="$1"
    local name
    name="$(config_name "$spec")"
    cp "$RESULT_ROOT/artifacts/$name/host_code" bin/host_code
    cp "$RESULT_ROOT/artifacts/$name/dpu_code" bin/dpu_code
}

echo "==> Building ${#configs[@]} configurations before collection"
for spec in "${configs[@]}"; do
    nr_dpus="${spec%%:*}"
    remainder="${spec#*:}"
    tasklets="${remainder%%:*}"
    remainder="${remainder#*:}"
    rank_paths="${remainder%%:*}"
    sysfs_ranks="${remainder#*:}"
    name="$(config_name "$spec")"
    result_dir="$RESULT_ROOT/$name"
    artifact_dir="$RESULT_ROOT/artifacts/$name"
    mkdir -p "$result_dir" "$artifact_dir"

    make clean
    make NR_DPUS="$nr_dpus" NR_TASKLETS="$tasklets" all \
        > "$result_dir/build.log" 2>&1
    cp bin/host_code bin/dpu_code "$artifact_dir/"
    sha256sum "$artifact_dir/host_code" "$artifact_dir/dpu_code" \
        > "$result_dir/binaries.sha256"
    printf 'NR_DPUS=%s\nNR_TASKLETS=%s\nNUMA_NODE=%s\nEXCLUDED_SYSFS_RANKS=%s\nSPMV_DPU_RANK_PATHS=%s\nEXPECTED_SYSFS_RANKS=%s\nN_WARMUP=%s\nN_REPS=%s\nTRANSPORT_KEY_VERSION=%s\nCOLLECTION_ORDER=LATIN_ROTATION\nMATRIX=data/bcsstk30.mtx\n' \
        "$nr_dpus" "$tasklets" "$NUMA_NODE" "$EXCLUDED_SYSFS_RANKS" \
        "$rank_paths" "$sysfs_ranks" "$N_WARMUP" "$N_REPS" \
        "$TRANSPORT_KEY_VERSION" > "$result_dir/config.txt"
done

run_rounds() {
    local phase="$1"
    local rounds="$2"
    local config_count="${#configs[@]}"
    local rep slot config_index spec name result_dir rep_id log_path
    local remainder rank_paths start_wall_ns end_wall_ns

    for rep in $(seq 1 "$rounds"); do
        shift_index=$(( (rep - 1) % config_count ))
        for slot in $(seq 0 $((config_count - 1))); do
            config_index=$(( (shift_index + slot) % config_count ))
            spec="${configs[$config_index]}"
            name="$(config_name "$spec")"
            result_dir="$RESULT_ROOT/$name"
            rep_id="$(printf '%02d' "$rep")"
            remainder="${spec#*:}"
            remainder="${remainder#*:}"
            rank_paths="${remainder%%:*}"
            activate_config "$spec"
            export SPMV_DPU_RANK_PATHS="$rank_paths"

            unset SPMV_TRACE_CSV SPMV_TRACE_RUN_ID SPMV_TRACE_REPEAT_ID \
                SPMV_TRACE_HOST_NUMA_NODE SPMV_TRACE_PROCESS_STATE \
                SPMV_TRACE_PREWARM_RUNS || true
            if [[ "$phase" == "trace" ]]; then
                export SPMV_TRACE_CSV="$result_dir/trace_${rep_id}.csv"
                export SPMV_TRACE_RUN_ID="$name"
                export SPMV_TRACE_REPEAT_ID="$rep"
                export SPMV_TRACE_HOST_NUMA_NODE="$TRACE_HOST_NUMA_NODE"
                export SPMV_TRACE_PROCESS_STATE="interleaved_fresh_process"
                export SPMV_TRACE_PREWARM_RUNS="$N_WARMUP"
                log_path="$result_dir/run_${rep_id}.log"
            else
                log_path="$result_dir/warmup_${rep_id}.log"
            fi

            echo "==> phase=$phase round=$rep slot=$slot config=$name"
            start_wall_ns="$(date +%s%N)"
            run_spmv 0 > "$log_path" 2>&1
            end_wall_ns="$(date +%s%N)"
            require_correct_result "$log_path"
            printf '%s,%s,%s,%s,%s,%s\n' \
                "$phase" "$rep" "$slot" "$name" \
                "$start_wall_ns" "$end_wall_ns" \
                >> "$RESULT_ROOT/schedule.csv"
        done
    done
    unset SPMV_DPU_RANK_PATHS SPMV_TRACE_CSV SPMV_TRACE_RUN_ID \
        SPMV_TRACE_REPEAT_ID SPMV_TRACE_HOST_NUMA_NODE \
        SPMV_TRACE_PROCESS_STATE SPMV_TRACE_PREWARM_RUNS || true
}

run_rounds warmup "$N_WARMUP"
run_rounds trace "$N_REPS"

for spec in "${configs[@]}"; do
    name="$(config_name "$spec")"
    result_dir="$RESULT_ROOT/$name"
    remainder="${spec#*:}"
    remainder="${remainder#*:}"
    sysfs_ranks="${remainder#*:}"
    python3 "$SCRIPT_DIR/validate_hw_trace.py" \
        --expected-host-numa-node "$NUMA_NODE" \
        --expected-dpu-numa-node "$NUMA_NODE" \
        --expected-sysfs-ranks "$sysfs_ranks" \
        "$result_dir"/trace_*.csv > "$result_dir/validation.log"
    python3 "$SCRIPT_DIR/analyze_transport_keys.py" \
        --min-samples "$TRANSPORT_KEY_MIN_SAMPLES" \
        --min-traces "$TRANSPORT_KEY_MIN_TRACES" \
        --spread-threshold-pct "$TRANSPORT_KEY_SPREAD_THRESHOLD_PCT" \
        --cv-threshold-pct "$TRANSPORT_KEY_CV_THRESHOLD_PCT" \
        --output "$result_dir/transport_key_summary.csv" \
        "$result_dir"/trace_*.csv \
        > "$result_dir/transport_key_analysis.log"
    echo "==> PASS $name"
done

mapfile -t all_traces < <(find "$RESULT_ROOT" -mindepth 2 -maxdepth 2 \
    -type f -name 'trace_*.csv' | sort)
python3 "$SCRIPT_DIR/analyze_transport_keys.py" \
    --min-samples "$TRANSPORT_KEY_MIN_SAMPLES" \
    --min-traces "$TRANSPORT_KEY_MIN_TRACES" \
    --spread-threshold-pct "$TRANSPORT_KEY_SPREAD_THRESHOLD_PCT" \
    --cv-threshold-pct "$TRANSPORT_KEY_CV_THRESHOLD_PCT" \
    --output "$RESULT_ROOT/transport_key_summary_all.csv" \
    "${all_traces[@]}" > "$RESULT_ROOT/transport_key_analysis_all.log"
python3 "$SCRIPT_DIR/summarize_scale_results.py" "$RESULT_ROOT" \
    --output "$RESULT_ROOT/stability_by_scale.csv" \
    > "$RESULT_ROOT/stability_by_scale.log"

archive=""
if [[ "$CREATE_ARCHIVE" == "1" ]]; then
    archive="${RESULT_ROOT}.tar.gz"
    tar -czf "$archive" -C "$(dirname "$RESULT_ROOT")" \
        "$(basename "$RESULT_ROOT")"
fi
mkdir -p "$(dirname "$LATEST_RESULT_POINTER")"
printf '%s\n' "$RESULT_ROOT" | tee "$LATEST_RESULT_POINTER"
echo "Trace results: $RESULT_ROOT"
echo "Scale summary: $RESULT_ROOT/stability_by_scale.csv"
if [[ -n "$archive" ]]; then
    echo "Archive:       $archive"
else
    echo "Archive:       disabled"
fi
