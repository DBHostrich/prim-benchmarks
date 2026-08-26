#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
: "${RESULT_ROOT:?Set RESULT_ROOT to a persistent result directory}"

NUMA_NODE="${NUMA_NODE:-0}"
N_WARMUP_PROCESSES="${N_WARMUP_PROCESSES:-5}"
N_REPS_PROCESSES="${N_REPS_PROCESSES:-30}"
INPUT_ELEMENTS="${INPUT_ELEMENTS:-2621440}"
TASKLETS="${TASKLETS:-16}"
BLOCK_SIZE_LOG2="${BLOCK_SIZE_LOG2:-10}"
DPU_PROFILE="${VA_DPU_PROFILE:-}"
CREATE_ARCHIVE="${CREATE_ARCHIVE:-0}"

if [[ "$N_WARMUP_PROCESSES" != "5" || "$N_REPS_PROCESSES" != "30" ||
      "$INPUT_ELEMENTS" != "2621440" || "$TASKLETS" != "16" ||
      "$BLOCK_SIZE_LOG2" != "10" ]]; then
    echo "VA baseline requires 5 warmup processes, 30 samples, 2621440 elements, 16 tasklets, and BL=10" >&2
    exit 1
fi
if [[ -e "$RESULT_ROOT" ]]; then
    echo "RESULT_ROOT must name a new directory: $RESULT_ROOT" >&2
    exit 1
fi
mkdir -p "$RESULT_ROOT"
cd "$SCRIPT_DIR"

command -v numactl >/dev/null 2>&1 || {
    echo "numactl is required for pinned VA hardware validation" >&2
    exit 1
}

uname -a > "$RESULT_ROOT/uname.txt"
lscpu > "$RESULT_ROOT/lscpu.txt"
numactl --hardware > "$RESULT_ROOT/numa.txt" 2>&1
numactl --show > "$RESULT_ROOT/numactl_show.txt" 2>&1
git -C "$PRIM_ROOT" rev-parse HEAD > "$RESULT_ROOT/prim_git_commit.txt"
git -C "$PRIM_ROOT" status --short > "$RESULT_ROOT/prim_git_status.txt"
dpu-upmem-dpurte-clang --version > "$RESULT_ROOT/dpu_compiler_version.txt" 2>&1
dpu-pkg-config --cflags --libs dpu > "$RESULT_ROOT/dpu_sdk_flags.txt" 2>&1
sha256sum Makefile host/app.c dpu/task.c support/common.h support/params.h \
    validate_va_baseline_trace.py run_hw_baseline_validation.sh \
    > "$RESULT_ROOT/source.sha256"

printf '%s\n' \
    "INPUT_ELEMENTS=$INPUT_ELEMENTS" \
    "TASKLETS=$TASKLETS" \
    "BLOCK_SIZE_LOG2=$BLOCK_SIZE_LOG2" \
    "SCALING=strong" \
    "N_WARMUP_PROCESSES=$N_WARMUP_PROCESSES" \
    "N_REPS_PROCESSES=$N_REPS_PROCESSES" \
    "N_WARMUP_IN_PROCESS=1" \
    "N_REPS_IN_PROCESS=1" \
    "NUMA_NODE=$NUMA_NODE" \
    "VA_DPU_PROFILE=$DPU_PROFILE" \
    "VA_VALIDATION_INPUT=1" \
    > "$RESULT_ROOT/config.txt"

make clean
make NR_DPUS=1 NR_TASKLETS="$TASKLETS" BL="$BLOCK_SIZE_LOG2" \
    TYPE=INT32 ENERGY=0 VA_VALIDATION_INPUT=1 all \
    > "$RESULT_ROOT/build.log" 2>&1
sha256sum bin/host_code bin/dpu_code > "$RESULT_ROOT/binaries.sha256"

export VA_SOURCE_SHA256
export VA_HOST_BINARY_SHA256
export VA_DPU_BINARY_SHA256
export VA_SDK_VERSION
export VA_HOST_NUMA_NODE
VA_SOURCE_SHA256="$(sha256sum "$RESULT_ROOT/source.sha256" | awk '{print $1}')"
VA_HOST_BINARY_SHA256="$(sha256sum bin/host_code | awk '{print $1}')"
VA_DPU_BINARY_SHA256="$(sha256sum bin/dpu_code | awk '{print $1}')"
VA_SDK_VERSION="$(head -n 1 "$RESULT_ROOT/dpu_compiler_version.txt" | tr ',' ';')"
VA_HOST_NUMA_NODE="$NUMA_NODE"

run_va() {
    numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE" \
        ./bin/host_code -w 1 -e 1 -i "$INPUT_ELEMENTS" -x 1
}

require_correct_result() {
    local log_path="$1"
    grep -q '\[OK\] Outputs are equal' "$log_path"
    grep -q 'VA_CHECKSUM expected=' "$log_path"
}

for configuration in single rank; do
    result_dir="$RESULT_ROOT/$configuration"
    mkdir -p "$result_dir"
    export VA_ALLOCATION_MODE="$configuration"
    export VA_DPU_PROFILE="$DPU_PROFILE"

    unset VA_TRACE_CSV VA_TRACE_RUN_ID VA_TRACE_REPEAT_ID || true
    for rep in $(seq 1 "$N_WARMUP_PROCESSES"); do
        warmup_log="$result_dir/warmup_$(printf '%02d' "$rep").log"
        run_va > "$warmup_log" 2>&1
        require_correct_result "$warmup_log"
    done

    for rep in $(seq 1 "$N_REPS_PROCESSES"); do
        rep_id="$(printf '%02d' "$rep")"
        export VA_TRACE_CSV="$result_dir/trace_${rep_id}.csv"
        export VA_TRACE_RUN_ID="VA_${configuration}"
        export VA_TRACE_REPEAT_ID="$rep"
        run_log="$result_dir/run_${rep_id}.log"
        run_va > "$run_log" 2>&1
        require_correct_result "$run_log"
    done
done
unset VA_TRACE_CSV VA_TRACE_RUN_ID VA_TRACE_REPEAT_ID VA_ALLOCATION_MODE || true

mapfile -t traces < <(find "$RESULT_ROOT" -mindepth 2 -maxdepth 2 \
    -type f -name 'trace_[0-9][0-9].csv' | sort)
python3 "$SCRIPT_DIR/validate_va_baseline_trace.py" \
    --expected-reps "$N_REPS_PROCESSES" \
    --input-elements "$INPUT_ELEMENTS" \
    --tasklets "$TASKLETS" \
    --block-size-log2 "$BLOCK_SIZE_LOG2" \
    --provenance-root "$RESULT_ROOT" \
    --output-csv "$RESULT_ROOT/va_hw_events.csv" \
    --manifest "$RESULT_ROOT/va_hardware_manifest.json" \
    "${traces[@]}" | tee "$RESULT_ROOT/validation.log"

archive=""
if [[ "$CREATE_ARCHIVE" == "1" ]]; then
    archive="${RESULT_ROOT}.tar.gz"
    tar -czf "$archive" -C "$(dirname "$RESULT_ROOT")" "$(basename "$RESULT_ROOT")"
fi

echo "PASS VA hardware baseline collection"
echo "Result:  $RESULT_ROOT"
if [[ -n "$archive" ]]; then
    echo "Archive: $archive"
fi
