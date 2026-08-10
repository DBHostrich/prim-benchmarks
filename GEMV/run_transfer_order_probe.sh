#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-/tmp/bdang/gemv_transfer_order_$(date +%Y%m%d_%H%M%S)}"
LATEST_RESULT_POINTER="${LATEST_RESULT_POINTER:-/tmp/bdang/latest_gemv_transfer_order_path.txt}"
NUMA_NODE="${NUMA_NODE:-0}"
EXCLUDED_SYSFS_RANKS="${EXCLUDED_SYSFS_RANKS:-5}"
NR_DPUS="${NR_DPUS:-128}"
TASKLETS="${TASKLETS:-16}"
M_SIZE="${M_SIZE:-8192}"
N_SIZE="${N_SIZE:-8192}"
PROCESS_WARMUP_RUNS="${PROCESS_WARMUP_RUNS:-1}"
TRACE_RUNS="${TRACE_RUNS:-24}"
IN_PROCESS_WARMUP="${IN_PROCESS_WARMUP:-1}"
IN_PROCESS_REPS="${IN_PROCESS_REPS:-3}"
HEARTBEAT_PERIOD_US="${HEARTBEAT_PERIOD_US:-100}"
HEARTBEAT_CALIBRATION_SECONDS="${HEARTBEAT_CALIBRATION_SECONDS:-5}"
HEARTBEAT_CALIBRATION_PERCENTILE="${HEARTBEAT_CALIBRATION_PERCENTILE:-99}"
HEARTBEAT_MIN_THRESHOLD_US="${HEARTBEAT_MIN_THRESHOLD_US:-200}"
CREATE_ARCHIVE="${CREATE_ARCHIVE:-1}"
DPU_RANK_TOPOLOGY_TSV="${DPU_RANK_TOPOLOGY_TSV:-}"
HOST_CPU_ID="${HOST_CPU_ID:-}"
PROBE_CPU_ID="${PROBE_CPU_ID:-}"
TRANSPORT_KEY_VERSION="v9_unchanged_transfer_order_diagnostic_only"
ORDER_VARIANTS=(MATRIX_THEN_VECTOR VECTOR_THEN_MATRIX)
probe_pid=""

cleanup_probe() {
    if [[ -n "$probe_pid" ]] && kill -0 "$probe_pid" 2>/dev/null; then
        kill -TERM "$probe_pid" 2>/dev/null || true
        wait "$probe_pid" 2>/dev/null || true
    fi
    probe_pid=""
}

report_failed_result_root() {
    local status="$?"
    cleanup_probe
    if (( status != 0 )); then
        echo "FAILED result root retained at: $RESULT_ROOT" >&2
    fi
}
trap report_failed_result_root EXIT

if [[ -z "$DPU_RANK_TOPOLOGY_TSV" || ! -r "$DPU_RANK_TOPOLOGY_TSV" ]]; then
    echo "ERROR: set DPU_RANK_TOPOLOGY_TSV to a readable topology TSV" >&2
    exit 1
fi
if [[ "$NUMA_NODE" != "0" || "$NR_DPUS" != "128" ]]; then
    echo "ERROR: this control experiment requires NUMA node 0 and 128 DPUs" >&2
    exit 1
fi
if [[ "$M_SIZE" != "8192" || "$N_SIZE" != "8192" ]]; then
    echo "ERROR: the validator requires M_SIZE=N_SIZE=8192" >&2
    exit 1
fi
if [[ "$IN_PROCESS_WARMUP" != "1" || "$IN_PROCESS_REPS" != "3" ]]; then
    echo "ERROR: use one in-process warmup and three measured repetitions" >&2
    exit 1
fi
for command in numactl cc python3 make git lscpu sha256sum tar \
    dpu-upmem-dpurte-clang dpu-pkg-config; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "ERROR: required command is absent: $command" >&2
        exit 1
    fi
done

read -r auto_host_cpu auto_probe_cpu _ < <(
    python3 "$SCRIPT_DIR/select_host_probe_cpus.py" --numa-node "$NUMA_NODE"
)
HOST_CPU_ID="${HOST_CPU_ID:-$auto_host_cpu}"
PROBE_CPU_ID="${PROBE_CPU_ID:-$auto_probe_cpu}"
if [[ "$HOST_CPU_ID" == "$PROBE_CPU_ID" ]]; then
    echo "ERROR: workload and heartbeat probe CPUs must differ" >&2
    exit 1
fi

mkdir -p "$RESULT_ROOT/artifacts" "$RESULT_ROOT/build_logs"
# Publish the retained result directory before any hardware or validation step.
# This keeps post-failure inspection commands usable under `set -e`.
mkdir -p "$(dirname "$LATEST_RESULT_POINTER")"
printf '%s\n' "$RESULT_ROOT" > "$LATEST_RESULT_POINTER"
cd "$SCRIPT_DIR"
uname -a > "$RESULT_ROOT/uname.txt"
lscpu > "$RESULT_ROOT/lscpu.txt"
lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE > "$RESULT_ROOT/lscpu_extended.txt"
numactl --hardware > "$RESULT_ROOT/numa.txt" 2>&1
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

cc -std=c11 -O2 -Wall -Wextra -Werror \
    "$SCRIPT_DIR/tools/stall_probe.c" \
    -o "$RESULT_ROOT/artifacts/stall_probe"
sha256sum "$RESULT_ROOT/artifacts/stall_probe" \
    > "$RESULT_ROOT/stall_probe.sha256"

wait_for_probe_header() {
    local output_path="$1"
    local ready=0
    for _ in $(seq 1 100); do
        if [[ -s "$output_path" ]]; then
            ready=1
            break
        fi
        if ! kill -0 "$probe_pid" 2>/dev/null; then
            break
        fi
        sleep 0.01
    done
    if (( ready == 0 )); then
        echo "ERROR: heartbeat probe did not become ready" >&2
        cleanup_probe
        return 1
    fi
}

calibration_csv="$RESULT_ROOT/heartbeat_calibration.csv"
"$RESULT_ROOT/artifacts/stall_probe" \
    -c "$PROBE_CPU_ID" -o "$calibration_csv" \
    -p "$HEARTBEAT_PERIOD_US" -t 1 &
probe_pid="$!"
wait_for_probe_header "$calibration_csv"
sleep "$HEARTBEAT_CALIBRATION_SECONDS"
kill -TERM "$probe_pid"
wait "$probe_pid"
probe_pid=""
HEARTBEAT_THRESHOLD_US="$(python3 \
    "$SCRIPT_DIR/calibrate_heartbeat_threshold.py" "$calibration_csv" \
    --minimum-us "$HEARTBEAT_MIN_THRESHOLD_US" \
    --percentile "$HEARTBEAT_CALIBRATION_PERCENTILE" \
    --summary "$RESULT_ROOT/heartbeat_calibration_summary.csv")"
if [[ ! "$HEARTBEAT_THRESHOLD_US" =~ ^[0-9]+$ ]]; then
    echo "ERROR: calibrated heartbeat threshold is invalid" >&2
    exit 1
fi
printf 'HOST_CPU_ID=%s\nPROBE_CPU_ID=%s\nHEARTBEAT_PERIOD_US=%s\nHEARTBEAT_CALIBRATION_SECONDS=%s\nHEARTBEAT_CALIBRATION_PERCENTILE=%s\nHEARTBEAT_MIN_THRESHOLD_US=%s\nHEARTBEAT_THRESHOLD_US=%s\n' \
    "$HOST_CPU_ID" "$PROBE_CPU_ID" "$HEARTBEAT_PERIOD_US" \
    "$HEARTBEAT_CALIBRATION_SECONDS" "$HEARTBEAT_CALIBRATION_PERCENTILE" \
    "$HEARTBEAT_MIN_THRESHOLD_US" "$HEARTBEAT_THRESHOLD_US" \
    > "$RESULT_ROOT/heartbeat_config.txt"

rank_count=$((NR_DPUS / 64))
rank_paths="$(python3 "$SCRIPT_DIR/select_rank_paths.py" \
    "$GEMV_TRACE_DPU_RANK_TOPOLOGY_TSV" \
    --numa-node "$NUMA_NODE" --exclude-sysfs-ranks "$EXCLUDED_SYSFS_RANKS" \
    --rank-count "$rank_count" --field rank_path)"
sysfs_ranks="$(python3 "$SCRIPT_DIR/select_rank_paths.py" \
    "$GEMV_TRACE_DPU_RANK_TOPOLOGY_TSV" \
    --numa-node "$NUMA_NODE" --exclude-sysfs-ranks "$EXCLUDED_SYSFS_RANKS" \
    --rank-count "$rank_count" --field sysfs_rank_id)"
channels="$(python3 "$SCRIPT_DIR/select_rank_paths.py" \
    "$GEMV_TRACE_DPU_RANK_TOPOLOGY_TSV" \
    --numa-node "$NUMA_NODE" --exclude-sysfs-ranks "$EXCLUDED_SYSFS_RANKS" \
    --rank-count "$rank_count" --field channel)"

make clean
make NR_DPUS="$NR_DPUS" NR_TASKLETS="$TASKLETS" all \
    > "$RESULT_ROOT/build_logs/GEMV_${NR_DPUS}dpu_${TASKLETS}tl.log" 2>&1
cp bin/gemv_host bin/gemv_dpu "$RESULT_ROOT/artifacts/"
sha256sum "$RESULT_ROOT/artifacts/gemv_host" \
    "$RESULT_ROOT/artifacts/gemv_dpu" > "$RESULT_ROOT/artifacts/binaries.sha256"

config_name() {
    printf 'GEMV_%sdpu_%stl_%s' "$NR_DPUS" "$TASKLETS" "$1"
}

for order in "${ORDER_VARIANTS[@]}"; do
    name="$(config_name "$order")"
    result_dir="$RESULT_ROOT/$name"
    mkdir -p "$result_dir"
    printf 'NR_DPUS=%s\nNR_TASKLETS=%s\nM_SIZE=%s\nN_SIZE=%s\nNUMA_NODE=%s\nEXCLUDED_SYSFS_RANKS=%s\nGEMV_DPU_RANK_PATHS=%s\nEXPECTED_SYSFS_RANKS=%s\nEXPECTED_CHANNELS=%s\nPROCESS_WARMUP_RUNS=%s\nTRACE_RUNS=%s\nIN_PROCESS_WARMUP=%s\nIN_PROCESS_REPS=%s\nTRANSPORT_KEY_VERSION=%s\nHOST_BINDING_MODE=FIXED_CORE\nHOST_CPU_LIST=%s\nPROBE_CPU_ID=%s\nTRANSFER_ORDER_VARIANT=%s\nHEARTBEAT_THRESHOLD_US=%s\nTIMING_SCOPE=PUSH_ONLY\nCOLLECTION_ORDER=ROTATING_PAIR\n' \
        "$NR_DPUS" "$TASKLETS" "$M_SIZE" "$N_SIZE" "$NUMA_NODE" \
        "$EXCLUDED_SYSFS_RANKS" "$rank_paths" "$sysfs_ranks" "$channels" \
        "$PROCESS_WARMUP_RUNS" "$TRACE_RUNS" "$IN_PROCESS_WARMUP" \
        "$IN_PROCESS_REPS" "$TRANSPORT_KEY_VERSION" "$HOST_CPU_ID" \
        "$PROBE_CPU_ID" "$order" "$HEARTBEAT_THRESHOLD_US" \
        > "$result_dir/config.txt"
done

printf 'phase,round,slot,config,transfer_order_variant,start_wall_ns,end_wall_ns\n' \
    > "$RESULT_ROOT/schedule.csv"

run_gemv() {
    numactl --physcpubind="$HOST_CPU_ID" --membind="$NUMA_NODE" \
        ./bin/gemv_host -m "$M_SIZE" -n "$N_SIZE" \
        -w "$IN_PROCESS_WARMUP" -e "$IN_PROCESS_REPS"
}

run_gemv_with_heartbeat() {
    local heartbeat_path="$1"
    local log_path="$2"
    local status probe_status
    "$RESULT_ROOT/artifacts/stall_probe" \
        -c "$PROBE_CPU_ID" -o "$heartbeat_path" \
        -p "$HEARTBEAT_PERIOD_US" -t "$HEARTBEAT_THRESHOLD_US" &
    probe_pid="$!"
    wait_for_probe_header "$heartbeat_path"
    set +e
    run_gemv > "$log_path" 2>&1
    status="$?"
    kill -TERM "$probe_pid" 2>/dev/null
    wait "$probe_pid"
    probe_status="$?"
    probe_pid=""
    set -e
    if (( status != 0 )); then
        return "$status"
    fi
    if (( probe_status != 0 )); then
        echo "ERROR: heartbeat probe failed with status $probe_status" >&2
        return "$probe_status"
    fi
}

require_correct_result() {
    local log_path="$1"
    if ! grep -q "Outputs are equal" "$log_path" \
        || grep -q "Outputs differ" "$log_path"; then
        echo "ERROR: GEMV correctness check failed in $log_path" >&2
        exit 1
    fi
}

run_rounds() {
    local phase="$1"
    local rounds="$2"
    local rep slot config_index order name result_dir rep_id
    local log_path heartbeat_path start_wall_ns end_wall_ns
    for rep in $(seq 1 "$rounds"); do
        for slot in 0 1; do
            config_index=$(( (rep - 1 + slot) % 2 ))
            order="${ORDER_VARIANTS[$config_index]}"
            name="$(config_name "$order")"
            result_dir="$RESULT_ROOT/$name"
            rep_id="$(printf '%02d' "$rep")"
            export GEMV_DPU_RANK_PATHS="$rank_paths"
            export GEMV_TRANSFER_ORDER="$order"
            unset GEMV_TRACE_CSV GEMV_TRACE_DPUS_CSV GEMV_TRACE_RUN_ID \
                GEMV_TRACE_REPEAT_ID GEMV_TRACE_HOST_NUMA_NODE \
                GEMV_TRACE_PROCESS_STATE GEMV_TRACE_PREWARM_RUNS \
                GEMV_TRACE_HOST_BINDING_MODE GEMV_TRACE_HOST_CPU_LIST || true
            if [[ "$phase" == "trace" ]]; then
                export GEMV_TRACE_CSV="$result_dir/trace_${rep_id}.csv"
                export GEMV_TRACE_DPUS_CSV="$result_dir/trace_${rep_id}_dpus.csv"
                export GEMV_TRACE_RUN_ID="$name"
                export GEMV_TRACE_REPEAT_ID="$rep"
                export GEMV_TRACE_HOST_NUMA_NODE="$NUMA_NODE"
                export GEMV_TRACE_PROCESS_STATE="transfer_order_probe"
                export GEMV_TRACE_PREWARM_RUNS="$PROCESS_WARMUP_RUNS"
                export GEMV_TRACE_HOST_BINDING_MODE="FIXED_CORE"
                export GEMV_TRACE_HOST_CPU_LIST="$HOST_CPU_ID"
                log_path="$result_dir/run_${rep_id}.log"
                heartbeat_path="$result_dir/heartbeat_${rep_id}.csv"
            else
                log_path="$result_dir/warmup_${rep_id}.log"
                heartbeat_path=""
            fi
            echo "==> phase=$phase round=$rep slot=$slot order=$order"
            start_wall_ns="$(date +%s%N)"
            if [[ "$phase" == "trace" ]]; then
                run_gemv_with_heartbeat "$heartbeat_path" "$log_path"
            else
                run_gemv > "$log_path" 2>&1
            fi
            end_wall_ns="$(date +%s%N)"
            require_correct_result "$log_path"
            printf '%s,%s,%s,%s,%s,%s,%s\n' \
                "$phase" "$rep" "$slot" "$name" "$order" \
                "$start_wall_ns" "$end_wall_ns" >> "$RESULT_ROOT/schedule.csv"
        done
    done
}

run_rounds warmup "$PROCESS_WARMUP_RUNS"
run_rounds trace "$TRACE_RUNS"
unset GEMV_DPU_RANK_PATHS GEMV_TRANSFER_ORDER GEMV_TRACE_CSV \
    GEMV_TRACE_DPUS_CSV GEMV_TRACE_RUN_ID GEMV_TRACE_REPEAT_ID \
    GEMV_TRACE_HOST_NUMA_NODE GEMV_TRACE_PROCESS_STATE \
    GEMV_TRACE_PREWARM_RUNS GEMV_TRACE_HOST_BINDING_MODE \
    GEMV_TRACE_HOST_CPU_LIST || true

for order in "${ORDER_VARIANTS[@]}"; do
    name="$(config_name "$order")"
    result_dir="$RESULT_ROOT/$name"
    mapfile -t config_traces < <(find "$result_dir" -maxdepth 1 \
        -type f -name 'trace_*.csv' ! -name '*_dpus.csv' | sort)
    python3 "$SCRIPT_DIR/validate_hw_trace.py" \
        --expected-host-numa-node "$NUMA_NODE" \
        --expected-dpu-numa-node "$NUMA_NODE" \
        --expected-sysfs-ranks "$sysfs_ranks" \
        --expected-host-binding-mode "FIXED_CORE" \
        --expected-host-cpu-list "$HOST_CPU_ID" \
        --expected-transfer-order-variant "$order" \
        "${config_traces[@]}" > "$result_dir/validation.log"
    echo "==> PASS $name"
done

python3 "$SCRIPT_DIR/analyze_transfer_order_probe.py" "$RESULT_ROOT" \
    > "$RESULT_ROOT/transfer_order_analysis.log"

archive=""
if [[ "$CREATE_ARCHIVE" == "1" ]]; then
    archive="${RESULT_ROOT}.tar.gz"
    tar -czf "$archive" -C "$(dirname "$RESULT_ROOT")" \
        "$(basename "$RESULT_ROOT")"
fi
printf '%s\n' "$RESULT_ROOT" | tee "$LATEST_RESULT_POINTER"
echo "Trace results:          $RESULT_ROOT"
echo "Order comparison:       $RESULT_ROOT/transfer_order_analysis/transfer_order_comparison.csv"
echo "Heartbeat calibration:  $RESULT_ROOT/heartbeat_calibration_summary.csv"
if [[ -n "$archive" ]]; then
    echo "Archive:                $archive"
fi
