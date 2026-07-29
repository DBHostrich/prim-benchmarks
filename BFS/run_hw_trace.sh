#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULT_ROOT="${RESULT_ROOT:-$WORKSPACE_DIR/bfs_hw_trace_$(date +%Y%m%d_%H%M%S)}"
NUMA_NODE="${NUMA_NODE:-0}"
N_WARMUP="${N_WARMUP:-5}"
N_REPS="${N_REPS:-30}"
DPUS_LIST="${DPUS_LIST:-256 512}"
TASKLETS_LIST="${TASKLETS_LIST:-1 2 4 8 16}"
GRAPH_PATH="data/loc-gowalla_edges.txt"
EXPECTED_GRAPH_SHA256="418c002fd2f70d25d6561465ffc7b4a6f14f7e406856aaba8ddc327ac4de10e6"

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
        printf 'NR_DPUS=%s\nNR_TASKLETS=%s\nNUMA_NODE=%s\nGRAPH=%s\n' \
            "$nr_dpus" "$tasklets" "$NUMA_NODE" "$GRAPH_PATH" \
            > "$result_dir/config.txt"

        unset BFS_TRACE_CSV BFS_TRACE_RUN_ID BFS_TRACE_REPEAT_ID || true
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
            run_log="$result_dir/run_${rep_id}.log"
            run_bfs 0 > "$run_log" 2>&1
            require_correct_result "$run_log"
        done
        unset BFS_TRACE_CSV BFS_TRACE_RUN_ID BFS_TRACE_REPEAT_ID || true

        python3 "$SCRIPT_DIR/validate_hw_trace.py" "$result_dir"/trace_*.csv \
            > "$result_dir/validation.log"
        echo "==> PASS $config"
    done
done

archive="${RESULT_ROOT}.tar.gz"
tar -czf "$archive" -C "$(dirname "$RESULT_ROOT")" "$(basename "$RESULT_ROOT")"
printf '%s\n' "$RESULT_ROOT" | tee "$WORKSPACE_DIR/latest_bfs_trace_result_path.txt"
echo "Trace results: $RESULT_ROOT"
echo "Archive:       $archive"
