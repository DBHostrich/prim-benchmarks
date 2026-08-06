#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-$WORKSPACE_DIR/bfs_interleaved_scale_$(date +%Y%m%d_%H%M%S)}"
LATEST_RESULT_POINTER="${LATEST_RESULT_POINTER:-$WORKSPACE_DIR/latest_bfs_interleaved_scale_path.txt}"
NUMA_NODE="${NUMA_NODE:-0}"
N_WARMUP="${N_WARMUP:-3}"
N_REPS="${N_REPS:-15}"
TRANSPORT_KEY_MIN_SAMPLES="${TRANSPORT_KEY_MIN_SAMPLES:-10}"
TRANSPORT_KEY_MIN_TRACES="${TRANSPORT_KEY_MIN_TRACES:-10}"
TRANSPORT_KEY_SPREAD_THRESHOLD_PCT="${TRANSPORT_KEY_SPREAD_THRESHOLD_PCT:-25}"
TRANSPORT_KEY_CV_THRESHOLD_PCT="${TRANSPORT_KEY_CV_THRESHOLD_PCT:-25}"
CREATE_ARCHIVE="${CREATE_ARCHIVE:-0}"
DPUS_LIST="${DPUS_LIST:-64 128 256}"
TASKLETS_LIST="${TASKLETS_LIST:-1}"
GRAPH_PATH="data/loc-gowalla_edges.txt"
EXPECTED_GRAPH_SHA256="418c002fd2f70d25d6561465ffc7b4a6f14f7e406856aaba8ddc327ac4de10e6"
TRANSPORT_KEY_VERSION="v6_mux_pair_context"

mkdir -p "$RESULT_ROOT/artifacts"
cd "$SCRIPT_DIR"

actual_graph_sha256="$(sha256sum "$GRAPH_PATH" | awk '{print $1}')"
if [[ "$actual_graph_sha256" != "$EXPECTED_GRAPH_SHA256" ]]; then
    echo "ERROR: $GRAPH_PATH SHA256 is $actual_graph_sha256" >&2
    echo "       expected $EXPECTED_GRAPH_SHA256" >&2
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
sha256sum "$GRAPH_PATH" > "$RESULT_ROOT/graph.sha256"
dpu-upmem-dpurte-clang --version \
    > "$RESULT_ROOT/dpu_compiler_version.txt" 2>&1 || true
dpu-pkg-config --cflags --libs dpu \
    > "$RESULT_ROOT/dpu_sdk_flags.txt" 2>&1 || true

if command -v numactl >/dev/null 2>&1; then
    TRACE_HOST_NUMA_NODE="$NUMA_NODE"
else
    TRACE_HOST_NUMA_NODE="unbound"
fi

configs=()
for nr_dpus in $DPUS_LIST; do
    for tasklets in $TASKLETS_LIST; do
        configs+=("${nr_dpus}:${tasklets}")
    done
done
if (( ${#configs[@]} < 2 )); then
    echo "ERROR: interleaved collection requires at least two configurations" >&2
    exit 1
fi
printf 'phase,round,slot,config,start_wall_ns,end_wall_ns\n' \
    > "$RESULT_ROOT/schedule.csv"

require_correct_result() {
    local log_path="$1"
    if grep -q "Mismatch at node" "$log_path"; then
        echo "ERROR: BFS verification mismatch in $log_path" >&2
        exit 1
    fi
}

run_bfs() {
    local verbosity="$1"
    if command -v numactl >/dev/null 2>&1; then
        numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE" \
            ./bin/host_code -v "$verbosity" -f "$GRAPH_PATH"
    else
        ./bin/host_code -v "$verbosity" -f "$GRAPH_PATH"
    fi
}

config_name() {
    local config_spec="$1"
    local nr_dpus="${config_spec%%:*}"
    local tasklets="${config_spec##*:}"
    printf 'BFS_%sdpu_%stl' "$nr_dpus" "$tasklets"
}

activate_config() {
    local config_spec="$1"
    local name
    name="$(config_name "$config_spec")"
    cp "$RESULT_ROOT/artifacts/$name/host_code" bin/host_code
    cp "$RESULT_ROOT/artifacts/$name/dpu_code" bin/dpu_code
}

echo "==> Building ${#configs[@]} configurations before collection"
for config_spec in "${configs[@]}"; do
    nr_dpus="${config_spec%%:*}"
    tasklets="${config_spec##*:}"
    name="$(config_name "$config_spec")"
    result_dir="$RESULT_ROOT/$name"
    artifact_dir="$RESULT_ROOT/artifacts/$name"
    mkdir -p "$result_dir" "$artifact_dir"

    make clean
    make NR_DPUS="$nr_dpus" NR_TASKLETS="$tasklets" all \
        > "$result_dir/build.log" 2>&1
    cp bin/host_code bin/dpu_code "$artifact_dir/"
    sha256sum "$artifact_dir/host_code" "$artifact_dir/dpu_code" \
        > "$result_dir/binaries.sha256"
    printf 'NR_DPUS=%s\nNR_TASKLETS=%s\nNUMA_NODE=%s\nTRACE_HOST_NUMA_NODE=%s\nN_WARMUP=%s\nN_REPS=%s\nTRANSPORT_KEY_VERSION=%s\nCOLLECTION_ORDER=LATIN_ROTATION\nGRAPH=%s\n' \
        "$nr_dpus" "$tasklets" "$NUMA_NODE" "$TRACE_HOST_NUMA_NODE" \
        "$N_WARMUP" "$N_REPS" "$TRANSPORT_KEY_VERSION" "$GRAPH_PATH" \
        > "$result_dir/config.txt"
done

run_rounds() {
    local phase="$1"
    local rounds="$2"
    local config_count="${#configs[@]}"

    for rep in $(seq 1 "$rounds"); do
        shift_index=$(( (rep - 1) % config_count ))
        for slot in $(seq 0 $((config_count - 1))); do
            config_index=$(( (shift_index + slot) % config_count ))
            config_spec="${configs[$config_index]}"
            name="$(config_name "$config_spec")"
            result_dir="$RESULT_ROOT/$name"
            rep_id="$(printf '%02d' "$rep")"
            activate_config "$config_spec"

            unset BFS_TRACE_CSV BFS_TRACE_RUN_ID BFS_TRACE_REPEAT_ID \
                BFS_TRACE_HOST_NUMA_NODE BFS_TRACE_PROCESS_STATE \
                BFS_TRACE_PREWARM_RUNS || true
            if [[ "$phase" == "trace" ]]; then
                export BFS_TRACE_CSV="$result_dir/trace_${rep_id}.csv"
                export BFS_TRACE_RUN_ID="$name"
                export BFS_TRACE_REPEAT_ID="$rep"
                export BFS_TRACE_HOST_NUMA_NODE="$TRACE_HOST_NUMA_NODE"
                export BFS_TRACE_PROCESS_STATE="interleaved_fresh_process"
                export BFS_TRACE_PREWARM_RUNS="$N_WARMUP"
                log_path="$result_dir/run_${rep_id}.log"
            else
                log_path="$result_dir/warmup_${rep_id}.log"
            fi

            echo "==> phase=$phase round=$rep slot=$slot config=$name"
            start_wall_ns="$(date +%s%N)"
            run_bfs 0 > "$log_path" 2>&1
            end_wall_ns="$(date +%s%N)"
            require_correct_result "$log_path"
            printf '%s,%s,%s,%s,%s,%s\n' \
                "$phase" "$rep" "$slot" "$name" \
                "$start_wall_ns" "$end_wall_ns" \
                >> "$RESULT_ROOT/schedule.csv"
        done
    done
    unset BFS_TRACE_CSV BFS_TRACE_RUN_ID BFS_TRACE_REPEAT_ID \
        BFS_TRACE_HOST_NUMA_NODE BFS_TRACE_PROCESS_STATE \
        BFS_TRACE_PREWARM_RUNS || true
}

run_rounds warmup "$N_WARMUP"
run_rounds trace "$N_REPS"

for config_spec in "${configs[@]}"; do
    name="$(config_name "$config_spec")"
    result_dir="$RESULT_ROOT/$name"
    python3 "$SCRIPT_DIR/validate_hw_trace.py" "$result_dir"/trace_*.csv \
        > "$result_dir/validation.log"
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
    "${all_traces[@]}" \
    > "$RESULT_ROOT/transport_key_analysis_all.log"
python3 "$SCRIPT_DIR/evaluate_transport_keys.py" \
    --holdout-unit configuration \
    --summary-output "$RESULT_ROOT/config_holdout_summary.csv" \
    --per-holdout-output "$RESULT_ROOT/config_holdout_per_config.csv" \
    "${all_traces[@]}" \
    > "$RESULT_ROOT/config_holdout.log"

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
