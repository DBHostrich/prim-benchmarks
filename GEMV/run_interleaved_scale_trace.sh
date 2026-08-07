#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-/tmp/bdang/gemv_v8_numa0_$(date +%Y%m%d_%H%M%S)}"
LATEST_RESULT_POINTER="${LATEST_RESULT_POINTER:-/tmp/bdang/latest_gemv_v8_numa0_path.txt}"
NUMA_NODE="${NUMA_NODE:-0}"
EXCLUDED_SYSFS_RANKS="${EXCLUDED_SYSFS_RANKS:-5}"
DPUS_LIST="${DPUS_LIST:-64 128 256 512 1024 1216}"
TASKLETS_LIST="${TASKLETS_LIST:-16}"
M_SIZE="${M_SIZE:-8192}"
N_SIZE="${N_SIZE:-8192}"
PROCESS_WARMUP_RUNS="${PROCESS_WARMUP_RUNS:-3}"
TRACE_RUNS="${TRACE_RUNS:-30}"
IN_PROCESS_WARMUP="${IN_PROCESS_WARMUP:-1}"
IN_PROCESS_REPS="${IN_PROCESS_REPS:-3}"
TRANSPORT_KEY_MIN_SAMPLES="${TRANSPORT_KEY_MIN_SAMPLES:-20}"
TRANSPORT_KEY_MIN_TRACES="${TRANSPORT_KEY_MIN_TRACES:-20}"
TRANSPORT_KEY_SPREAD_THRESHOLD_PCT="${TRANSPORT_KEY_SPREAD_THRESHOLD_PCT:-25}"
TRANSPORT_KEY_CV_THRESHOLD_PCT="${TRANSPORT_KEY_CV_THRESHOLD_PCT:-25}"
CREATE_ARCHIVE="${CREATE_ARCHIVE:-0}"
DPU_RANK_TOPOLOGY_TSV="${DPU_RANK_TOPOLOGY_TSV:-}"
TRANSPORT_KEY_VERSION="v8_collection_physical_cpu_dpu_topology"

if [[ -z "$DPU_RANK_TOPOLOGY_TSV" || ! -r "$DPU_RANK_TOPOLOGY_TSV" ]]; then
    echo "ERROR: set DPU_RANK_TOPOLOGY_TSV to a readable dpu_rank_topology.tsv" >&2
    exit 1
fi
if [[ "$NUMA_NODE" != "0" ]]; then
    echo "ERROR: this experiment is defined for NUMA_NODE=0" >&2
    exit 1
fi
if [[ "$M_SIZE" != "8192" || "$N_SIZE" != "8192" ]]; then
    echo "ERROR: the validator defines GEMV with M_SIZE=8192 and N_SIZE=8192" >&2
    exit 1
fi
if [[ "$IN_PROCESS_WARMUP" != "1" || "$IN_PROCESS_REPS" != "3" ]]; then
    echo "ERROR: the validator expects one in-process warmup and three measured repetitions" >&2
    exit 1
fi
if ! command -v numactl >/dev/null 2>&1; then
    echo "ERROR: numactl is required for the NUMA0 experiment" >&2
    exit 1
fi

mkdir -p "$RESULT_ROOT/artifacts"
cd "$SCRIPT_DIR"

uname -a > "$RESULT_ROOT/uname.txt"
lscpu > "$RESULT_ROOT/lscpu.txt"
numactl --hardware > "$RESULT_ROOT/numa.txt" 2>&1
numactl --show > "$RESULT_ROOT/numactl_show.txt" 2>&1
ls -l /dev/dpu_rank* > "$RESULT_ROOT/dpu_rank_devices.txt" 2>&1
git -C "$WORKSPACE_DIR/prim-benchmarks" rev-parse HEAD \
    > "$RESULT_ROOT/prim_git_commit.txt"
git -C "$WORKSPACE_DIR/prim-benchmarks" status --short \
    > "$RESULT_ROOT/prim_git_status.txt"
dpu-upmem-dpurte-clang --version \
    > "$RESULT_ROOT/dpu_compiler_version.txt" 2>&1
dpu-pkg-config --cflags --libs dpu > "$RESULT_ROOT/dpu_sdk_flags.txt" 2>&1
cp "$DPU_RANK_TOPOLOGY_TSV" "$RESULT_ROOT/dpu_rank_topology.tsv"
sha256sum "$RESULT_ROOT/dpu_rank_topology.tsv" \
    > "$RESULT_ROOT/dpu_rank_topology.sha256"
export GEMV_TRACE_DPU_RANK_TOPOLOGY_TSV="$RESULT_ROOT/dpu_rank_topology.tsv"

configs=()
for nr_dpus in $DPUS_LIST; do
    if (( nr_dpus % 64 != 0 )); then
        echo "ERROR: DPU count $nr_dpus is not a whole number of ranks" >&2
        exit 1
    fi
    rank_count=$((nr_dpus / 64))
    rank_paths="$(python3 "$SCRIPT_DIR/select_rank_paths.py" \
        "$GEMV_TRACE_DPU_RANK_TOPOLOGY_TSV" \
        --numa-node "$NUMA_NODE" \
        --exclude-sysfs-ranks "$EXCLUDED_SYSFS_RANKS" \
        --rank-count "$rank_count" --field rank_path)"
    sysfs_ranks="$(python3 "$SCRIPT_DIR/select_rank_paths.py" \
        "$GEMV_TRACE_DPU_RANK_TOPOLOGY_TSV" \
        --numa-node "$NUMA_NODE" \
        --exclude-sysfs-ranks "$EXCLUDED_SYSFS_RANKS" \
        --rank-count "$rank_count" --field sysfs_rank_id)"
    channels="$(python3 "$SCRIPT_DIR/select_rank_paths.py" \
        "$GEMV_TRACE_DPU_RANK_TOPOLOGY_TSV" \
        --numa-node "$NUMA_NODE" \
        --exclude-sysfs-ranks "$EXCLUDED_SYSFS_RANKS" \
        --rank-count "$rank_count" --field channel)"
    for tasklets in $TASKLETS_LIST; do
        configs+=("${nr_dpus}:${tasklets}:${rank_paths}:${sysfs_ranks}:${channels}")
    done
done
if (( ${#configs[@]} < 2 )); then
    echo "ERROR: interleaved collection requires at least two configurations" >&2
    exit 1
fi
printf 'phase,round,slot,config,start_wall_ns,end_wall_ns\n' \
    > "$RESULT_ROOT/schedule.csv"

config_name() {
    local spec="$1"
    local nr_dpus tasklets remainder
    nr_dpus="${spec%%:*}"
    remainder="${spec#*:}"
    tasklets="${remainder%%:*}"
    printf 'GEMV_%sdpu_%stl' "$nr_dpus" "$tasklets"
}

activate_config() {
    local spec="$1"
    local name
    name="$(config_name "$spec")"
    cp "$RESULT_ROOT/artifacts/$name/gemv_host" bin/gemv_host
    cp "$RESULT_ROOT/artifacts/$name/gemv_dpu" bin/gemv_dpu
}

run_gemv() {
    numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE" \
        ./bin/gemv_host -m "$M_SIZE" -n "$N_SIZE" \
        -w "$IN_PROCESS_WARMUP" -e "$IN_PROCESS_REPS"
}

require_correct_result() {
    local log_path="$1"
    if ! grep -q "Outputs are equal" "$log_path"; then
        echo "ERROR: GEMV verification marker is absent in $log_path" >&2
        exit 1
    fi
    if grep -q "Outputs differ" "$log_path"; then
        echo "ERROR: GEMV verification mismatch in $log_path" >&2
        exit 1
    fi
}

echo "==> Building ${#configs[@]} GEMV configurations before collection"
for spec in "${configs[@]}"; do
    nr_dpus="${spec%%:*}"
    remainder="${spec#*:}"
    tasklets="${remainder%%:*}"
    remainder="${remainder#*:}"
    rank_paths="${remainder%%:*}"
    remainder="${remainder#*:}"
    sysfs_ranks="${remainder%%:*}"
    channels="${remainder#*:}"
    name="$(config_name "$spec")"
    result_dir="$RESULT_ROOT/$name"
    artifact_dir="$RESULT_ROOT/artifacts/$name"
    mkdir -p "$result_dir" "$artifact_dir"

    make clean
    make NR_DPUS="$nr_dpus" NR_TASKLETS="$tasklets" all \
        > "$result_dir/build.log" 2>&1
    cp bin/gemv_host bin/gemv_dpu "$artifact_dir/"
    sha256sum "$artifact_dir/gemv_host" "$artifact_dir/gemv_dpu" \
        > "$result_dir/binaries.sha256"
    printf 'NR_DPUS=%s\nNR_TASKLETS=%s\nM_SIZE=%s\nN_SIZE=%s\nNUMA_NODE=%s\nEXCLUDED_SYSFS_RANKS=%s\nGEMV_DPU_RANK_PATHS=%s\nEXPECTED_SYSFS_RANKS=%s\nEXPECTED_CHANNELS=%s\nPROCESS_WARMUP_RUNS=%s\nTRACE_RUNS=%s\nIN_PROCESS_WARMUP=%s\nIN_PROCESS_REPS=%s\nTRANSPORT_KEY_VERSION=%s\nTIMING_SCOPE=PUSH_ONLY\nCOLLECTION_ORDER=LATIN_ROTATION\n' \
        "$nr_dpus" "$tasklets" "$M_SIZE" "$N_SIZE" "$NUMA_NODE" \
        "$EXCLUDED_SYSFS_RANKS" "$rank_paths" "$sysfs_ranks" "$channels" \
        "$PROCESS_WARMUP_RUNS" "$TRACE_RUNS" "$IN_PROCESS_WARMUP" \
        "$IN_PROCESS_REPS" "$TRANSPORT_KEY_VERSION" \
        > "$result_dir/config.txt"
done

run_rounds() {
    local phase="$1"
    local rounds="$2"
    local config_count="${#configs[@]}"
    local rep slot shift_index config_index spec name result_dir rep_id
    local remainder rank_paths log_path start_wall_ns end_wall_ns

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
            export GEMV_DPU_RANK_PATHS="$rank_paths"

            unset GEMV_TRACE_CSV GEMV_TRACE_DPUS_CSV GEMV_TRACE_RUN_ID \
                GEMV_TRACE_REPEAT_ID GEMV_TRACE_HOST_NUMA_NODE \
                GEMV_TRACE_PROCESS_STATE GEMV_TRACE_PREWARM_RUNS || true
            if [[ "$phase" == "trace" ]]; then
                export GEMV_TRACE_CSV="$result_dir/trace_${rep_id}.csv"
                export GEMV_TRACE_DPUS_CSV="$result_dir/trace_${rep_id}_dpus.csv"
                export GEMV_TRACE_RUN_ID="$name"
                export GEMV_TRACE_REPEAT_ID="$rep"
                export GEMV_TRACE_HOST_NUMA_NODE="$NUMA_NODE"
                export GEMV_TRACE_PROCESS_STATE="interleaved_fresh_process"
                export GEMV_TRACE_PREWARM_RUNS="$PROCESS_WARMUP_RUNS"
                log_path="$result_dir/run_${rep_id}.log"
            else
                log_path="$result_dir/warmup_${rep_id}.log"
            fi

            echo "==> phase=$phase round=$rep slot=$slot config=$name"
            start_wall_ns="$(date +%s%N)"
            run_gemv > "$log_path" 2>&1
            end_wall_ns="$(date +%s%N)"
            require_correct_result "$log_path"
            printf '%s,%s,%s,%s,%s,%s\n' \
                "$phase" "$rep" "$slot" "$name" \
                "$start_wall_ns" "$end_wall_ns" \
                >> "$RESULT_ROOT/schedule.csv"
        done
    done
    unset GEMV_DPU_RANK_PATHS GEMV_TRACE_CSV GEMV_TRACE_DPUS_CSV \
        GEMV_TRACE_RUN_ID GEMV_TRACE_REPEAT_ID GEMV_TRACE_HOST_NUMA_NODE \
        GEMV_TRACE_PROCESS_STATE GEMV_TRACE_PREWARM_RUNS || true
}

run_rounds warmup "$PROCESS_WARMUP_RUNS"
run_rounds trace "$TRACE_RUNS"

for spec in "${configs[@]}"; do
    name="$(config_name "$spec")"
    result_dir="$RESULT_ROOT/$name"
    remainder="${spec#*:}"
    remainder="${remainder#*:}"
    remainder="${remainder#*:}"
    sysfs_ranks="${remainder%%:*}"
    mapfile -t config_traces < <(find "$result_dir" -maxdepth 1 \
        -type f -name 'trace_*.csv' ! -name '*_dpus.csv' | sort)
    python3 "$SCRIPT_DIR/validate_hw_trace.py" \
        --expected-host-numa-node "$NUMA_NODE" \
        --expected-dpu-numa-node "$NUMA_NODE" \
        --expected-sysfs-ranks "$sysfs_ranks" \
        "${config_traces[@]}" > "$result_dir/validation.log"
    python3 "$SCRIPT_DIR/analyze_transport_keys.py" \
        --min-samples "$TRANSPORT_KEY_MIN_SAMPLES" \
        --min-traces "$TRANSPORT_KEY_MIN_TRACES" \
        --spread-threshold-pct "$TRANSPORT_KEY_SPREAD_THRESHOLD_PCT" \
        --cv-threshold-pct "$TRANSPORT_KEY_CV_THRESHOLD_PCT" \
        --output "$result_dir/transport_key_summary.csv" \
        "${config_traces[@]}" \
        > "$result_dir/transport_key_analysis.log"
    echo "==> PASS $name"
done

mapfile -t all_traces < <(find "$RESULT_ROOT" -mindepth 2 -maxdepth 2 \
    -type f -name 'trace_*.csv' ! -name '*_dpus.csv' | sort)
python3 "$SCRIPT_DIR/analyze_transport_keys.py" \
    --min-samples "$TRANSPORT_KEY_MIN_SAMPLES" \
    --min-traces "$TRANSPORT_KEY_MIN_TRACES" \
    --spread-threshold-pct "$TRANSPORT_KEY_SPREAD_THRESHOLD_PCT" \
    --cv-threshold-pct "$TRANSPORT_KEY_CV_THRESHOLD_PCT" \
    --output "$RESULT_ROOT/transport_key_summary_all.csv" \
    "${all_traces[@]}" > "$RESULT_ROOT/transport_key_analysis_all.log"
python3 "$SCRIPT_DIR/summarize_scale_stability.py" \
    "$RESULT_ROOT/transport_key_summary_all.csv" \
    --output "$RESULT_ROOT/scale_stability_summary.csv"

archive=""
if [[ "$CREATE_ARCHIVE" == "1" ]]; then
    archive="${RESULT_ROOT}.tar.gz"
    tar -czf "$archive" -C "$(dirname "$RESULT_ROOT")" \
        "$(basename "$RESULT_ROOT")"
fi
mkdir -p "$(dirname "$LATEST_RESULT_POINTER")"
printf '%s\n' "$RESULT_ROOT" | tee "$LATEST_RESULT_POINTER"
echo "Trace results:    $RESULT_ROOT"
echo "Scale summary:    $RESULT_ROOT/scale_stability_summary.csv"
if [[ -n "$archive" ]]; then
    echo "Archive:          $archive"
else
    echo "Archive:          disabled"
fi
