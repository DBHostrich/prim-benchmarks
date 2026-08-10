#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-/tmp/bdang/gemv_runtime_stall_$(date +%Y%m%d_%H%M%S)}"
LATEST_RESULT_POINTER="${LATEST_RESULT_POINTER:-/tmp/bdang/latest_gemv_runtime_stall_path.txt}"
NUMA_NODE="${NUMA_NODE:-0}"
EXCLUDED_SYSFS_RANKS="${EXCLUDED_SYSFS_RANKS:-5}"
DPUS_LIST="${DPUS_LIST:-128 1024}"
TASKLETS="${TASKLETS:-16}"
M_SIZE="${M_SIZE:-8192}"
N_SIZE="${N_SIZE:-8192}"
PROCESS_WARMUP_RUNS="${PROCESS_WARMUP_RUNS:-1}"
TRACE_RUNS="${TRACE_RUNS:-12}"
IN_PROCESS_WARMUP="${IN_PROCESS_WARMUP:-1}"
IN_PROCESS_REPS="${IN_PROCESS_REPS:-3}"
HEARTBEAT_PERIOD_US="${HEARTBEAT_PERIOD_US:-100}"
HEARTBEAT_THRESHOLD_US="${HEARTBEAT_THRESHOLD_US:-200}"
HEARTBEAT_WINDOW_NS="${HEARTBEAT_WINDOW_NS:-200000}"
CREATE_ARCHIVE="${CREATE_ARCHIVE:-1}"
DPU_RANK_TOPOLOGY_TSV="${DPU_RANK_TOPOLOGY_TSV:-}"
HOST_CPU_ID="${HOST_CPU_ID:-}"
PROBE_CPU_ID="${PROBE_CPU_ID:-}"
MULTI_CORE_CPU_LIST="${MULTI_CORE_CPU_LIST:-}"
TRANSPORT_KEY_VERSION="v9_unchanged_runtime_diagnostics_only"
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
if [[ "$NUMA_NODE" != "0" ]]; then
    echo "ERROR: this diagnostic experiment is defined for NUMA node 0" >&2
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

read -r auto_host_cpu auto_probe_cpu auto_multi_cpu_list < <(
    python3 "$SCRIPT_DIR/select_host_probe_cpus.py" --numa-node "$NUMA_NODE"
)
HOST_CPU_ID="${HOST_CPU_ID:-$auto_host_cpu}"
PROBE_CPU_ID="${PROBE_CPU_ID:-$auto_probe_cpu}"
MULTI_CORE_CPU_LIST="${MULTI_CORE_CPU_LIST:-$auto_multi_cpu_list}"
if [[ "$HOST_CPU_ID" == "$PROBE_CPU_ID" ]]; then
    echo "ERROR: workload and heartbeat probe CPUs must differ" >&2
    exit 1
fi
if [[ ",${MULTI_CORE_CPU_LIST}," == *",${PROBE_CPU_ID},"* ]]; then
    echo "ERROR: MULTI_CORE_CPU_LIST must exclude PROBE_CPU_ID" >&2
    exit 1
fi
if [[ ",${MULTI_CORE_CPU_LIST}," != *",${HOST_CPU_ID},"* ]]; then
    echo "ERROR: MULTI_CORE_CPU_LIST must contain HOST_CPU_ID" >&2
    exit 1
fi

mkdir -p "$RESULT_ROOT/artifacts" "$RESULT_ROOT/build_logs"
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
printf 'HOST_CPU_ID=%s\nPROBE_CPU_ID=%s\nMULTI_CORE_CPU_LIST=%s\nHEARTBEAT_PERIOD_US=%s\nHEARTBEAT_THRESHOLD_US=%s\nHEARTBEAT_WINDOW_NS=%s\n' \
    "$HOST_CPU_ID" "$PROBE_CPU_ID" "$MULTI_CORE_CPU_LIST" \
    "$HEARTBEAT_PERIOD_US" "$HEARTBEAT_THRESHOLD_US" \
    "$HEARTBEAT_WINDOW_NS" > "$RESULT_ROOT/runtime_probe_config.txt"

cc -std=c11 -O2 -Wall -Wextra -Werror \
    "$SCRIPT_DIR/tools/stall_probe.c" \
    -o "$RESULT_ROOT/artifacts/stall_probe"
sha256sum "$RESULT_ROOT/artifacts/stall_probe" \
    > "$RESULT_ROOT/stall_probe.sha256"

configs=()
for nr_dpus in $DPUS_LIST; do
    if (( nr_dpus % 64 != 0 )); then
        echo "ERROR: DPU count $nr_dpus is not a whole number of ranks" >&2
        exit 1
    fi
    rank_count=$((nr_dpus / 64))
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
    artifact_name="GEMV_${nr_dpus}dpu_${TASKLETS}tl"
    artifact_dir="$RESULT_ROOT/artifacts/$artifact_name"
    mkdir -p "$artifact_dir"
    make clean
    make NR_DPUS="$nr_dpus" NR_TASKLETS="$TASKLETS" all \
        > "$RESULT_ROOT/build_logs/${artifact_name}.log" 2>&1
    cp bin/gemv_host bin/gemv_dpu "$artifact_dir/"
    sha256sum "$artifact_dir/gemv_host" "$artifact_dir/gemv_dpu" \
        > "$artifact_dir/binaries.sha256"
    for binding in MULTI_CORE FIXED_CORE; do
        configs+=("${nr_dpus};${binding};${rank_paths};${sysfs_ranks};${channels}")
    done
done
if (( ${#configs[@]} != 4 )); then
    echo "ERROR: use exactly two DPU scales, producing four interleaved configs" >&2
    exit 1
fi

config_name() {
    local spec="$1"
    local nr_dpus binding unused
    IFS=';' read -r nr_dpus binding unused <<< "$spec"
    printf 'GEMV_%sdpu_%stl_%s' "$nr_dpus" "$TASKLETS" "$binding"
}

cpu_list_for_binding() {
    local binding="$1"
    if [[ "$binding" == "FIXED_CORE" ]]; then
        printf '%s' "$HOST_CPU_ID"
    elif [[ "$binding" == "MULTI_CORE" ]]; then
        printf '%s' "$MULTI_CORE_CPU_LIST"
    else
        echo "ERROR: unknown binding mode $binding" >&2
        return 1
    fi
}

for spec in "${configs[@]}"; do
    IFS=';' read -r nr_dpus binding rank_paths sysfs_ranks channels <<< "$spec"
    name="$(config_name "$spec")"
    result_dir="$RESULT_ROOT/$name"
    cpu_list="$(cpu_list_for_binding "$binding")"
    mkdir -p "$result_dir"
    printf 'NR_DPUS=%s\nNR_TASKLETS=%s\nM_SIZE=%s\nN_SIZE=%s\nNUMA_NODE=%s\nEXCLUDED_SYSFS_RANKS=%s\nGEMV_DPU_RANK_PATHS=%s\nEXPECTED_SYSFS_RANKS=%s\nEXPECTED_CHANNELS=%s\nPROCESS_WARMUP_RUNS=%s\nTRACE_RUNS=%s\nIN_PROCESS_WARMUP=%s\nIN_PROCESS_REPS=%s\nTRANSPORT_KEY_VERSION=%s\nHOST_BINDING_MODE=%s\nHOST_CPU_LIST=%s\nPROBE_CPU_ID=%s\nTIMING_SCOPE=PUSH_ONLY\nCOLLECTION_ORDER=LATIN_ROTATION\n' \
        "$nr_dpus" "$TASKLETS" "$M_SIZE" "$N_SIZE" "$NUMA_NODE" \
        "$EXCLUDED_SYSFS_RANKS" "$rank_paths" "$sysfs_ranks" "$channels" \
        "$PROCESS_WARMUP_RUNS" "$TRACE_RUNS" "$IN_PROCESS_WARMUP" \
        "$IN_PROCESS_REPS" "$TRANSPORT_KEY_VERSION" "$binding" \
        "$cpu_list" "$PROBE_CPU_ID" > "$result_dir/config.txt"
done

printf 'phase,round,slot,config,binding_mode,host_cpu_list,start_wall_ns,end_wall_ns\n' \
    > "$RESULT_ROOT/schedule.csv"

activate_config() {
    local nr_dpus="$1"
    local artifact_name="GEMV_${nr_dpus}dpu_${TASKLETS}tl"
    cp "$RESULT_ROOT/artifacts/$artifact_name/gemv_host" bin/gemv_host
    cp "$RESULT_ROOT/artifacts/$artifact_name/gemv_dpu" bin/gemv_dpu
}

run_gemv() {
    local cpu_list="$1"
    numactl --physcpubind="$cpu_list" --membind="$NUMA_NODE" \
        ./bin/gemv_host -m "$M_SIZE" -n "$N_SIZE" \
        -w "$IN_PROCESS_WARMUP" -e "$IN_PROCESS_REPS"
}

run_gemv_with_heartbeat() {
    local cpu_list="$1"
    local heartbeat_path="$2"
    local log_path="$3"
    local status probe_status ready=0
    "$RESULT_ROOT/artifacts/stall_probe" \
        -c "$PROBE_CPU_ID" -o "$heartbeat_path" \
        -p "$HEARTBEAT_PERIOD_US" -t "$HEARTBEAT_THRESHOLD_US" &
    probe_pid="$!"
    for _ in $(seq 1 100); do
        if [[ -s "$heartbeat_path" ]]; then
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
    set +e
    run_gemv "$cpu_list" > "$log_path" 2>&1
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
    local config_count="${#configs[@]}"
    local rep slot shift_index config_index spec name result_dir rep_id
    local nr_dpus binding rank_paths sysfs_ranks channels cpu_list
    local log_path heartbeat_path start_wall_ns end_wall_ns

    for rep in $(seq 1 "$rounds"); do
        shift_index=$(( (rep - 1) % config_count ))
        for slot in $(seq 0 $((config_count - 1))); do
            config_index=$(( (shift_index + slot) % config_count ))
            spec="${configs[$config_index]}"
            IFS=';' read -r nr_dpus binding rank_paths sysfs_ranks channels <<< "$spec"
            name="$(config_name "$spec")"
            result_dir="$RESULT_ROOT/$name"
            rep_id="$(printf '%02d' "$rep")"
            cpu_list="$(cpu_list_for_binding "$binding")"
            activate_config "$nr_dpus"
            export GEMV_DPU_RANK_PATHS="$rank_paths"
            unset GEMV_TRACE_CSV GEMV_TRACE_DPUS_CSV GEMV_TRACE_RUN_ID \
                GEMV_TRACE_REPEAT_ID GEMV_TRACE_HOST_NUMA_NODE \
                GEMV_TRACE_PROCESS_STATE GEMV_TRACE_PREWARM_RUNS \
                GEMV_TRACE_HOST_BINDING_MODE GEMV_TRACE_HOST_CPU_LIST \
                GEMV_TRANSFER_ORDER || true
            if [[ "$phase" == "trace" ]]; then
                export GEMV_TRACE_CSV="$result_dir/trace_${rep_id}.csv"
                export GEMV_TRACE_DPUS_CSV="$result_dir/trace_${rep_id}_dpus.csv"
                export GEMV_TRACE_RUN_ID="$name"
                export GEMV_TRACE_REPEAT_ID="$rep"
                export GEMV_TRACE_HOST_NUMA_NODE="$NUMA_NODE"
                export GEMV_TRACE_PROCESS_STATE="runtime_stall_probe"
                export GEMV_TRACE_PREWARM_RUNS="$PROCESS_WARMUP_RUNS"
                export GEMV_TRACE_HOST_BINDING_MODE="$binding"
                export GEMV_TRACE_HOST_CPU_LIST="$cpu_list"
                export GEMV_TRANSFER_ORDER="MATRIX_THEN_VECTOR"
                log_path="$result_dir/run_${rep_id}.log"
                heartbeat_path="$result_dir/heartbeat_${rep_id}.csv"
            else
                log_path="$result_dir/warmup_${rep_id}.log"
                heartbeat_path=""
            fi

            echo "==> phase=$phase round=$rep slot=$slot config=$name cpu=$cpu_list"
            start_wall_ns="$(date +%s%N)"
            if [[ "$phase" == "trace" ]]; then
                run_gemv_with_heartbeat "$cpu_list" "$heartbeat_path" "$log_path"
            else
                run_gemv "$cpu_list" > "$log_path" 2>&1
            fi
            end_wall_ns="$(date +%s%N)"
            require_correct_result "$log_path"
            printf '%s,%s,%s,%s,%s,"%s",%s,%s\n' \
                "$phase" "$rep" "$slot" "$name" "$binding" "$cpu_list" \
                "$start_wall_ns" "$end_wall_ns" >> "$RESULT_ROOT/schedule.csv"
        done
    done
}

run_rounds warmup "$PROCESS_WARMUP_RUNS"
run_rounds trace "$TRACE_RUNS"
unset GEMV_DPU_RANK_PATHS GEMV_TRACE_CSV GEMV_TRACE_DPUS_CSV \
    GEMV_TRACE_RUN_ID GEMV_TRACE_REPEAT_ID GEMV_TRACE_HOST_NUMA_NODE \
    GEMV_TRACE_PROCESS_STATE GEMV_TRACE_PREWARM_RUNS \
    GEMV_TRACE_HOST_BINDING_MODE GEMV_TRACE_HOST_CPU_LIST \
    GEMV_TRANSFER_ORDER || true

for spec in "${configs[@]}"; do
    IFS=';' read -r nr_dpus binding rank_paths sysfs_ranks channels <<< "$spec"
    name="$(config_name "$spec")"
    result_dir="$RESULT_ROOT/$name"
    cpu_list="$(cpu_list_for_binding "$binding")"
    mapfile -t config_traces < <(find "$result_dir" -maxdepth 1 \
        -type f -name 'trace_*.csv' ! -name '*_dpus.csv' | sort)
    python3 "$SCRIPT_DIR/validate_hw_trace.py" \
        --expected-host-numa-node "$NUMA_NODE" \
        --expected-dpu-numa-node "$NUMA_NODE" \
        --expected-sysfs-ranks "$sysfs_ranks" \
        --expected-host-binding-mode "$binding" \
        --expected-host-cpu-list "$cpu_list" \
        --expected-transfer-order-variant "MATRIX_THEN_VECTOR" \
        "${config_traces[@]}" > "$result_dir/validation.log"
    echo "==> PASS $name"
done

python3 "$SCRIPT_DIR/analyze_runtime_stalls.py" "$RESULT_ROOT" \
    --heartbeat-window-ns "$HEARTBEAT_WINDOW_NS" \
    > "$RESULT_ROOT/runtime_stall_analysis.log"

archive=""
if [[ "$CREATE_ARCHIVE" == "1" ]]; then
    archive="${RESULT_ROOT}.tar.gz"
    tar -czf "$archive" -C "$(dirname "$RESULT_ROOT")" \
        "$(basename "$RESULT_ROOT")"
fi
mkdir -p "$(dirname "$LATEST_RESULT_POINTER")"
printf '%s\n' "$RESULT_ROOT" | tee "$LATEST_RESULT_POINTER"
echo "Trace results:          $RESULT_ROOT"
echo "Runtime stall summary:  $RESULT_ROOT/runtime_stall_analysis/runtime_stall_summary.csv"
if [[ -n "$archive" ]]; then
    echo "Archive:                $archive"
fi
