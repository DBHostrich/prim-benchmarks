#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-/dev/shm/bfs_api_order_probe_$(date +%Y%m%d_%H%M%S)}"
LATEST_RESULT_POINTER="${LATEST_RESULT_POINTER:-/dev/shm/latest_bfs_api_order_probe_path.txt}"
NUMA_NODE="${NUMA_NODE:-0}"
NR_DPUS="${NR_DPUS:-256}"
NR_TASKLETS="${NR_TASKLETS:-1}"
N_PROCESS_REPS="${N_PROCESS_REPS:-5}"
SAMPLES_PER_PROCESS="${SAMPLES_PER_PROCESS:-20}"
WARMUPS_PER_PROCESS="${WARMUPS_PER_PROCESS:-3}"
EFFECT_THRESHOLD_PCT="${EFFECT_THRESHOLD_PCT:-5}"
MIN_SAMPLES="${MIN_SAMPLES:-20}"
MIN_TRACES="${MIN_TRACES:-5}"
SPREAD_THRESHOLD_PCT="${SPREAD_THRESHOLD_PCT:-25}"
CV_THRESHOLD_PCT="${CV_THRESHOLD_PCT:-25}"
GRAPH_PATH="data/loc-gowalla_edges.txt"
EXPECTED_GRAPH_SHA256="418c002fd2f70d25d6561465ffc7b4a6f14f7e406856aaba8ddc327ac4de10e6"

case "$RESULT_ROOT" in
    /dev/shm/*) ;;
    *)
        echo "ERROR: RESULT_ROOT must be under /dev/shm: $RESULT_ROOT" >&2
        exit 1
        ;;
esac

mkdir -p "$RESULT_ROOT"
cd "$SCRIPT_DIR"

actual_graph_sha256="$(sha256sum "$GRAPH_PATH" | awk '{print $1}')"
if [[ "$actual_graph_sha256" != "$EXPECTED_GRAPH_SHA256" ]]; then
    echo "ERROR: $GRAPH_PATH SHA256 is $actual_graph_sha256" >&2
    echo "       expected $EXPECTED_GRAPH_SHA256" >&2
    exit 1
fi

uname -a > "$RESULT_ROOT/uname.txt"
lscpu > "$RESULT_ROOT/lscpu.txt"
df -h /dev/shm > "$RESULT_ROOT/dev_shm_space.txt"
numactl --hardware > "$RESULT_ROOT/numa.txt" 2>&1 || true
numactl --show > "$RESULT_ROOT/numactl_show.txt" 2>&1 || true
ls -l /dev/dpu_rank* > "$RESULT_ROOT/dpu_rank_devices.txt" 2>&1 || true
git -C "$WORKSPACE_DIR/prim-benchmarks" rev-parse HEAD > "$RESULT_ROOT/prim_git_commit.txt"
git -C "$WORKSPACE_DIR/prim-benchmarks" rev-parse --abbrev-ref HEAD > "$RESULT_ROOT/prim_git_branch.txt"
git -C "$WORKSPACE_DIR/prim-benchmarks" status --short > "$RESULT_ROOT/prim_git_status.txt"
sha256sum "$GRAPH_PATH" > "$RESULT_ROOT/graph.sha256"
dpu-upmem-dpurte-clang --version > "$RESULT_ROOT/dpu_compiler_version.txt" 2>&1
dpu-pkg-config --cflags --libs dpu > "$RESULT_ROOT/dpu_sdk_flags.txt" 2>&1

printf '%s\n' \
    "NR_DPUS=$NR_DPUS" \
    "NR_TASKLETS=$NR_TASKLETS" \
    "NUMA_NODE=$NUMA_NODE" \
    "N_PROCESS_REPS=$N_PROCESS_REPS" \
    "SAMPLES_PER_PROCESS=$SAMPLES_PER_PROCESS" \
    "WARMUPS_PER_PROCESS=$WARMUPS_PER_PROCESS" \
    "EFFECT_THRESHOLD_PCT=$EFFECT_THRESHOLD_PCT" \
    "MIN_SAMPLES=$MIN_SAMPLES" \
    "MIN_TRACES=$MIN_TRACES" \
    "SPREAD_THRESHOLD_PCT=$SPREAD_THRESHOLD_PCT" \
    "CV_THRESHOLD_PCT=$CV_THRESHOLD_PCT" \
    "GRAPH_PATH=$GRAPH_PATH" \
    > "$RESULT_ROOT/config.txt"

echo "==> Building BFS API-order probe: ${NR_DPUS} DPUs, ${NR_TASKLETS} tasklet(s)"
make clean
make NR_DPUS="$NR_DPUS" NR_TASKLETS="$NR_TASKLETS" all \
    > "$RESULT_ROOT/build.log" 2>&1
sha256sum bin/host_code bin/dpu_code > "$RESULT_ROOT/binaries.sha256"

run_probe() {
    if command -v numactl >/dev/null 2>&1; then
        numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE" \
            ./bin/host_code -v 0 -f "$GRAPH_PATH"
    else
        ./bin/host_code -v 0 -f "$GRAPH_PATH"
    fi
}

unset BFS_TRACE_CSV BFS_TRACE_RUN_ID BFS_TRACE_REPEAT_ID \
    BFS_TRACE_HOST_NUMA_NODE BFS_TRACE_PROCESS_STATE \
    BFS_TRACE_PREWARM_RUNS BFS_CONTEXT_PROBE_CSV \
    BFS_CONTEXT_PROBE_RUN_ID BFS_CONTEXT_PROBE_PROCESS_REPEAT \
    BFS_CONTEXT_PROBE_SAMPLES BFS_CONTEXT_PROBE_WARMUPS \
    BFS_CONTEXT_PROBE_DPU BFS_GROUP_CONTEXT_PROBE_CSV \
    BFS_GROUP_CONTEXT_PROBE_RUN_ID BFS_GROUP_CONTEXT_PROBE_PROCESS_REPEAT \
    BFS_GROUP_CONTEXT_PROBE_SAMPLES BFS_GROUP_CONTEXT_PROBE_WARMUPS || true

for process_rep in $(seq 1 "$N_PROCESS_REPS"); do
    rep_id="$(printf '%02d' "$process_rep")"
    export BFS_API_ORDER_PROBE_CSV="$RESULT_ROOT/probe_${rep_id}.csv"
    export BFS_API_ORDER_PROBE_RUN_ID="BFS_${NR_DPUS}dpu_${NR_TASKLETS}tl_api_order"
    export BFS_API_ORDER_PROBE_PROCESS_REPEAT="$process_rep"
    export BFS_API_ORDER_PROBE_SAMPLES="$SAMPLES_PER_PROCESS"
    export BFS_API_ORDER_PROBE_WARMUPS="$WARMUPS_PER_PROCESS"
    run_log="$RESULT_ROOT/run_${rep_id}.log"
    echo "==> fresh process $process_rep/$N_PROCESS_REPS"
    run_probe > "$run_log" 2>&1
    grep -q \
        "API-order probe wrote $SAMPLES_PER_PROCESS samples per condition" \
        "$run_log"
done

unset BFS_API_ORDER_PROBE_CSV BFS_API_ORDER_PROBE_RUN_ID \
    BFS_API_ORDER_PROBE_PROCESS_REPEAT BFS_API_ORDER_PROBE_SAMPLES \
    BFS_API_ORDER_PROBE_WARMUPS || true

mapfile -t traces < <(
    find "$RESULT_ROOT" -maxdepth 1 -type f -name 'probe_*.csv' | sort
)
if [[ "${#traces[@]}" -eq 0 ]]; then
    echo "ERROR: API-order probe produced zero CSV traces" >&2
    exit 1
fi

python3 "$SCRIPT_DIR/analyze_api_order_probe.py" \
    --effect-threshold-pct "$EFFECT_THRESHOLD_PCT" \
    --min-samples "$MIN_SAMPLES" \
    --min-traces "$MIN_TRACES" \
    --spread-threshold-pct "$SPREAD_THRESHOLD_PCT" \
    --cv-threshold-pct "$CV_THRESHOLD_PCT" \
    --per-dpu-summary-output "$RESULT_ROOT/per_dpu_condition_summary.csv" \
    --per-dpu-paired-output "$RESULT_ROOT/per_dpu_paired_effects.csv" \
    --group-summary-output "$RESULT_ROOT/group_condition_summary.csv" \
    --group-paired-output "$RESULT_ROOT/group_paired_effects.csv" \
    "${traces[@]}" \
    | tee "$RESULT_ROOT/analysis.log"

printf '%s\n' "$RESULT_ROOT" | tee "$LATEST_RESULT_POINTER"
echo "API-order probe results: $RESULT_ROOT"
