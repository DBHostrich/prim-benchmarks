#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-$WORKSPACE_DIR/bfs_hw_trace_$(date +%Y%m%d_%H%M%S)}"
LATEST_RESULT_POINTER="${LATEST_RESULT_POINTER:-$WORKSPACE_DIR/latest_bfs_trace_result_path.txt}"
NUMA_NODE="${NUMA_NODE:-0}"
N_WARMUP="${N_WARMUP:-5}"
N_REPS="${N_REPS:-30}"
TRANSPORT_KEY_MIN_SAMPLES="${TRANSPORT_KEY_MIN_SAMPLES:-20}"
TRANSPORT_KEY_MIN_TRACES="${TRANSPORT_KEY_MIN_TRACES:-20}"
TRANSPORT_KEY_SPREAD_THRESHOLD_PCT="${TRANSPORT_KEY_SPREAD_THRESHOLD_PCT:-25}"
TRANSPORT_KEY_CV_THRESHOLD_PCT="${TRANSPORT_KEY_CV_THRESHOLD_PCT:-25}"
CREATE_ARCHIVE="${CREATE_ARCHIVE:-1}"
DPUS_LIST="${DPUS_LIST:-256 512}"
TASKLETS_LIST="${TASKLETS_LIST:-1 2 4 8 16}"
GRAPH_PATH="data/loc-gowalla_edges.txt"
EXPECTED_GRAPH_SHA256="418c002fd2f70d25d6561465ffc7b4a6f14f7e406856aaba8ddc327ac4de10e6"
TRANSPORT_KEY_VERSION="v4_history_min"

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
numactl --hardware > "$RESULT_ROOT/numa.txt" 2>&1 || true
numactl --show > "$RESULT_ROOT/numactl_show.txt" 2>&1 || true
git -C "$WORKSPACE_DIR/prim-benchmarks" rev-parse HEAD > "$RESULT_ROOT/prim_git_commit.txt"
git -C "$WORKSPACE_DIR/prim-benchmarks" status --short > "$RESULT_ROOT/prim_git_status.txt"
sha256sum "$GRAPH_PATH" > "$RESULT_ROOT/graph.sha256"
dpu-upmem-dpurte-clang --version > "$RESULT_ROOT/dpu_compiler_version.txt" 2>&1 || true
dpu-pkg-config --cflags --libs dpu > "$RESULT_ROOT/dpu_sdk_flags.txt" 2>&1 || true

run_bfs() {
    local verbosity="$1"
    if command -v numactl >/dev/null 2>&1; then
        numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE" \
            ./bin/host_code -v "$verbosity" -f "$GRAPH_PATH"
    else
        ./bin/host_code -v "$verbosity" -f "$GRAPH_PATH"
    fi
}

if command -v numactl >/dev/null 2>&1; then
    TRACE_HOST_NUMA_NODE="$NUMA_NODE"
else
    TRACE_HOST_NUMA_NODE="unbound"
fi

require_correct_result() {
    local log_path="$1"
    if grep -q "Mismatch at node" "$log_path"; then
        echo "ERROR: BFS verification mismatch in $log_path" >&2
        exit 1
    fi
}

for nr_dpus in $DPUS_LIST; do
    for tasklets in $TASKLETS_LIST; do
        config="BFS_${nr_dpus}dpu_${tasklets}tl"
        result_dir="$RESULT_ROOT/$config"
        mkdir -p "$result_dir"

        echo "==> Building $config"
        make clean
        make NR_DPUS="$nr_dpus" NR_TASKLETS="$tasklets" all \
            > "$result_dir/build.log" 2>&1
        sha256sum bin/host_code bin/dpu_code > "$result_dir/binaries.sha256"
        printf 'NR_DPUS=%s\nNR_TASKLETS=%s\nNUMA_NODE=%s\nTRACE_HOST_NUMA_NODE=%s\nN_WARMUP=%s\nN_REPS=%s\nTRANSPORT_KEY_VERSION=%s\nTRANSPORT_KEY_MIN_SAMPLES=%s\nTRANSPORT_KEY_MIN_TRACES=%s\nTRANSPORT_KEY_SPREAD_THRESHOLD_PCT=%s\nTRANSPORT_KEY_CV_THRESHOLD_PCT=%s\nCREATE_ARCHIVE=%s\nGRAPH=%s\n' \
            "$nr_dpus" "$tasklets" "$NUMA_NODE" "$TRACE_HOST_NUMA_NODE" \
            "$N_WARMUP" "$N_REPS" "$TRANSPORT_KEY_VERSION" \
            "$TRANSPORT_KEY_MIN_SAMPLES" \
            "$TRANSPORT_KEY_MIN_TRACES" \
            "$TRANSPORT_KEY_SPREAD_THRESHOLD_PCT" \
            "$TRANSPORT_KEY_CV_THRESHOLD_PCT" "$CREATE_ARCHIVE" \
            "$GRAPH_PATH" \
            > "$result_dir/config.txt"

        unset BFS_TRACE_CSV BFS_TRACE_RUN_ID BFS_TRACE_REPEAT_ID \
            BFS_TRACE_HOST_NUMA_NODE BFS_TRACE_PROCESS_STATE \
            BFS_TRACE_PREWARM_RUNS || true
        echo "==> Warming up $config ($N_WARMUP runs, tracing disabled)"
        for rep in $(seq 1 "$N_WARMUP"); do
            verbosity=0
            if [[ "$rep" -eq 1 ]]; then
                verbosity=1
            fi
            warmup_log="$result_dir/warmup_$(printf '%02d' "$rep").log"
            run_bfs "$verbosity" > "$warmup_log" 2>&1
            require_correct_result "$warmup_log"
        done

        echo "==> Tracing $config ($N_REPS runs)"
        for rep in $(seq 1 "$N_REPS"); do
            rep_id="$(printf '%02d' "$rep")"
            export BFS_TRACE_CSV="$result_dir/trace_${rep_id}.csv"
            export BFS_TRACE_RUN_ID="$config"
            export BFS_TRACE_REPEAT_ID="$rep"
            export BFS_TRACE_HOST_NUMA_NODE="$TRACE_HOST_NUMA_NODE"
            export BFS_TRACE_PROCESS_STATE="fresh_process"
            export BFS_TRACE_PREWARM_RUNS="$N_WARMUP"
            run_log="$result_dir/run_${rep_id}.log"
            run_bfs 0 > "$run_log" 2>&1
            require_correct_result "$run_log"
        done
        unset BFS_TRACE_CSV BFS_TRACE_RUN_ID BFS_TRACE_REPEAT_ID \
            BFS_TRACE_HOST_NUMA_NODE BFS_TRACE_PROCESS_STATE \
            BFS_TRACE_PREWARM_RUNS || true

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
        echo "==> PASS $config"
    done
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
