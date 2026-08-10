#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-/tmp/bdang/gemv_vector_replay_$(date +%Y%m%d_%H%M%S)}"
LATEST_RESULT_POINTER="${LATEST_RESULT_POINTER:-/tmp/bdang/latest_gemv_vector_replay_path.txt}"
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
TRANSFER_ORDER_VARIANT="MATRIX_THEN_VECTOR"
VECTOR_REPLAY_MODE="IDENTICAL_REPLAY"
TRANSPORT_KEY_VERSION="v9_unchanged_vector_replay_diagnostic_only"
CONFIG_NAME="GEMV_128dpu_16tl_VECTOR_REPLAY"
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
    echo "ERROR: this probe requires NUMA node 0 and 128 DPUs" >&2
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
(
    cd "$RESULT_ROOT"
    sha256sum dpu_rank_topology.tsv > dpu_rank_topology.sha256
)
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

result_dir="$RESULT_ROOT/$CONFIG_NAME"
mkdir -p "$result_dir"
printf 'NR_DPUS=%s\nNR_TASKLETS=%s\nM_SIZE=%s\nN_SIZE=%s\nNUMA_NODE=%s\nEXCLUDED_SYSFS_RANKS=%s\nGEMV_DPU_RANK_PATHS=%s\nEXPECTED_SYSFS_RANKS=%s\nEXPECTED_CHANNELS=%s\nPROCESS_WARMUP_RUNS=%s\nTRACE_RUNS=%s\nIN_PROCESS_WARMUP=%s\nIN_PROCESS_REPS=%s\nTRANSPORT_KEY_VERSION=%s\nHOST_BINDING_MODE=FIXED_CORE\nHOST_CPU_LIST=%s\nPROBE_CPU_ID=%s\nTRANSFER_ORDER_VARIANT=%s\nVECTOR_REPLAY_MODE=%s\nHEARTBEAT_THRESHOLD_US=%s\nTIMING_SCOPE=PUSH_ONLY\n' \
    "$NR_DPUS" "$TASKLETS" "$M_SIZE" "$N_SIZE" "$NUMA_NODE" \
    "$EXCLUDED_SYSFS_RANKS" "$rank_paths" "$sysfs_ranks" "$channels" \
    "$PROCESS_WARMUP_RUNS" "$TRACE_RUNS" "$IN_PROCESS_WARMUP" \
    "$IN_PROCESS_REPS" "$TRANSPORT_KEY_VERSION" "$HOST_CPU_ID" \
    "$PROBE_CPU_ID" "$TRANSFER_ORDER_VARIANT" "$VECTOR_REPLAY_MODE" \
    "$HEARTBEAT_THRESHOLD_US" > "$result_dir/config.txt"
printf 'phase,repeat,config,start_wall_ns,end_wall_ns\n' \
    > "$RESULT_ROOT/schedule.csv"

export GEMV_DPU_RANK_PATHS="$rank_paths"
export GEMV_TRANSFER_ORDER="$TRANSFER_ORDER_VARIANT"
export GEMV_VECTOR_REPLAY_MODE="$VECTOR_REPLAY_MODE"

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

unset_trace_env() {
    unset GEMV_TRACE_CSV GEMV_TRACE_DPUS_CSV GEMV_TRACE_RUN_ID \
        GEMV_TRACE_REPEAT_ID GEMV_TRACE_HOST_NUMA_NODE \
        GEMV_TRACE_PROCESS_STATE GEMV_TRACE_PREWARM_RUNS \
        GEMV_TRACE_HOST_BINDING_MODE GEMV_TRACE_HOST_CPU_LIST || true
}

for rep in $(seq 1 "$PROCESS_WARMUP_RUNS"); do
    unset_trace_env
    rep_id="$(printf '%02d' "$rep")"
    log_path="$result_dir/warmup_${rep_id}.log"
    start_wall_ns="$(date +%s%N)"
    echo "==> phase=warmup repeat=$rep"
    run_gemv > "$log_path" 2>&1
    end_wall_ns="$(date +%s%N)"
    require_correct_result "$log_path"
    printf 'warmup,%s,%s,%s,%s\n' "$rep" "$CONFIG_NAME" \
        "$start_wall_ns" "$end_wall_ns" >> "$RESULT_ROOT/schedule.csv"
done

for rep in $(seq 1 "$TRACE_RUNS"); do
    rep_id="$(printf '%02d' "$rep")"
    export GEMV_TRACE_CSV="$result_dir/trace_${rep_id}.csv"
    export GEMV_TRACE_DPUS_CSV="$result_dir/trace_${rep_id}_dpus.csv"
    export GEMV_TRACE_RUN_ID="$CONFIG_NAME"
    export GEMV_TRACE_REPEAT_ID="$rep"
    export GEMV_TRACE_HOST_NUMA_NODE="$NUMA_NODE"
    export GEMV_TRACE_PROCESS_STATE="vector_replay_probe"
    export GEMV_TRACE_PREWARM_RUNS="$PROCESS_WARMUP_RUNS"
    export GEMV_TRACE_HOST_BINDING_MODE="FIXED_CORE"
    export GEMV_TRACE_HOST_CPU_LIST="$HOST_CPU_ID"
    log_path="$result_dir/run_${rep_id}.log"
    heartbeat_path="$result_dir/heartbeat_${rep_id}.csv"
    start_wall_ns="$(date +%s%N)"
    echo "==> phase=trace repeat=$rep"
    run_gemv_with_heartbeat "$heartbeat_path" "$log_path"
    end_wall_ns="$(date +%s%N)"
    require_correct_result "$log_path"
    printf 'trace,%s,%s,%s,%s\n' "$rep" "$CONFIG_NAME" \
        "$start_wall_ns" "$end_wall_ns" >> "$RESULT_ROOT/schedule.csv"
done
unset_trace_env
unset GEMV_DPU_RANK_PATHS GEMV_TRANSFER_ORDER GEMV_VECTOR_REPLAY_MODE || true

mapfile -t traces < <(find "$result_dir" -maxdepth 1 \
    -type f -name 'trace_*.csv' ! -name '*_dpus.csv' | sort)
python3 "$SCRIPT_DIR/validate_hw_trace.py" \
    --expected-host-numa-node "$NUMA_NODE" \
    --expected-dpu-numa-node "$NUMA_NODE" \
    --expected-sysfs-ranks "$sysfs_ranks" \
    --expected-host-binding-mode "FIXED_CORE" \
    --expected-host-cpu-list "$HOST_CPU_ID" \
    --expected-transfer-order-variant "$TRANSFER_ORDER_VARIANT" \
    --expected-vector-replay-mode "$VECTOR_REPLAY_MODE" \
    "${traces[@]}" > "$result_dir/validation.log"

python3 "$SCRIPT_DIR/analyze_vector_replay_probe.py" "$RESULT_ROOT" \
    > "$RESULT_ROOT/vector_replay_analysis.log"

archive=""
if [[ "$CREATE_ARCHIVE" == "1" ]]; then
    archive="${RESULT_ROOT}.tar.gz"
    tar -czf "$archive" -C "$(dirname "$RESULT_ROOT")" \
        "$(basename "$RESULT_ROOT")"
fi
printf '%s\n' "$RESULT_ROOT" | tee "$LATEST_RESULT_POINTER"
echo "Trace results:      $RESULT_ROOT"
echo "Replay comparison:  $RESULT_ROOT/vector_replay_analysis/vector_replay_comparison.csv"
echo "Heartbeat config:   $RESULT_ROOT/heartbeat_calibration_summary.csv"
if [[ -n "$archive" ]]; then
    echo "Archive:            $archive"
fi
