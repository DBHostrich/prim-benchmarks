#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
: "${RESULT_ROOT:?Set RESULT_ROOT below /tmp}"
: "${SDK_SOURCE_ROOT:?Set SDK_SOURCE_ROOT to the UPMEM SDK 2025.1.0 source root}"

case "$RESULT_ROOT" in
    /tmp/*) ;;
    *)
        echo "RESULT_ROOT must be below /tmp: $RESULT_ROOT" >&2
        exit 1
        ;;
esac
if [[ -e "$RESULT_ROOT" ]]; then
    echo "RESULT_ROOT must name a new path: $RESULT_ROOT" >&2
    exit 1
fi

NUMA_NODE="${NUMA_NODE:-0}"
COLLECTION_MODE="${VA_TRANSFER_COLLECTION_MODE:-anchor}"
N_WARMUP_PROCESSES="${N_WARMUP_PROCESSES:-5}"
N_REPS_PROCESSES="${N_REPS_PROCESSES:-30}"
N_OVERHEAD_PROCESSES="${N_OVERHEAD_PROCESSES:-10}"
INPUT_ELEMENTS="${INPUT_ELEMENTS:-2621440}"
TASKLETS="${TASKLETS:-16}"
BLOCK_SIZE_LOG2="${BLOCK_SIZE_LOG2:-10}"
if [[ "$COLLECTION_MODE" == "sweep" ]]; then
    DPU_PROFILE="${VA_DPU_PROFILE:-backend=hw,regionMode=perf}"
else
    DPU_PROFILE="${VA_DPU_PROFILE:-backend=hw}"
fi
SWEEP_INPUT_ELEMENTS_TEXT="${VA_SWEEP_INPUT_ELEMENTS:-8192 16384 32768 131072 524288 1048576 2621440 4194304 8388608}"
SWEEP_OVERHEAD_ELEMENTS_TEXT="${VA_SWEEP_OVERHEAD_ELEMENTS:-8192 2621440 8388608}"
BUILD_JOBS="${BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN)}"
SDK_VERSION="2025.1.0"
SDK_HASH_MANIFEST="$SCRIPT_DIR/sdk/upmem-2025.1.0-source.sha256"
SDK_PATCH="$SCRIPT_DIR/sdk/upmem-2025.1.0-transfer-trace.patch"

stage() {
    printf '[VA] %s\n' "$1"
}

case "$COLLECTION_MODE" in
    anchor)
        if [[ "$N_WARMUP_PROCESSES" != "5" || "$N_REPS_PROCESSES" != "30" ||
              "$INPUT_ELEMENTS" != "2621440" || "$TASKLETS" != "16" ||
              "$BLOCK_SIZE_LOG2" != "10" ]]; then
            echo "Formal VA collection requires 5 warmup processes, 30 samples, 2621440 elements, 16 tasklets, and BL=10" >&2
            exit 1
        fi
        ;;
    sweep)
        if [[ "$N_WARMUP_PROCESSES" != "5" || "$N_REPS_PROCESSES" != "30" ||
              "$N_OVERHEAD_PROCESSES" != "10" || "$TASKLETS" != "16" ||
              "$BLOCK_SIZE_LOG2" != "10" ]]; then
            echo "Formal VA sweep requires 5 warmups, 30 samples, 10 overhead samples, 16 tasklets, and BL=10" >&2
            exit 1
        fi
        ;;
    *)
        echo "VA_TRANSFER_COLLECTION_MODE must be anchor or sweep: $COLLECTION_MODE" >&2
        exit 1
        ;;
esac
if (( N_OVERHEAD_PROCESSES < 2 )); then
    echo "N_OVERHEAD_PROCESSES must be at least 2" >&2
    exit 1
fi
case ",$DPU_PROFILE," in
    *,backend=hw,*) ;;
    *)
        echo "VA_DPU_PROFILE must select backend=hw for hardware acceptance: $DPU_PROFILE" >&2
        exit 1
        ;;
esac
if [[ "$COLLECTION_MODE" == "sweep" ]]; then
    case ",$DPU_PROFILE," in
        *,regionMode=perf,*) ;;
        *)
            echo "VA_DPU_PROFILE must select regionMode=perf for the size sweep: $DPU_PROFILE" >&2
            exit 1
            ;;
    esac
fi
read -r -a SWEEP_INPUT_ELEMENTS <<< "$SWEEP_INPUT_ELEMENTS_TEXT"
read -r -a SWEEP_OVERHEAD_ELEMENTS <<< "$SWEEP_OVERHEAD_ELEMENTS_TEXT"
if [[ "$COLLECTION_MODE" == "sweep" ]]; then
    declare -A sweep_seen=()
    for input_elements in "${SWEEP_INPUT_ELEMENTS[@]}"; do
        if [[ ! "$input_elements" =~ ^[0-9]+$ ]] || (( input_elements <= 0 )) ||
           (( input_elements % 64 != 0 )) || (( (input_elements / 64) % 2 != 0 )); then
            echo "invalid sweep input size: $input_elements" >&2
            exit 1
        fi
        if [[ -n "${sweep_seen[$input_elements]+present}" ]]; then
            echo "duplicate sweep input size: $input_elements" >&2
            exit 1
        fi
        sweep_seen[$input_elements]=1
    done
    for input_elements in "${SWEEP_OVERHEAD_ELEMENTS[@]}"; do
        if [[ -z "${sweep_seen[$input_elements]+present}" ]]; then
            echo "overhead input size is outside the sweep: $input_elements" >&2
            exit 1
        fi
    done
    if [[ -z "${sweep_seen[$INPUT_ELEMENTS]+present}" ]]; then
        echo "anchor INPUT_ELEMENTS must be part of the sweep: $INPUT_ELEMENTS" >&2
        exit 1
    fi
fi
for command_name in cmake patch perf numactl sha256sum readelf ldd python3 tar; do
    command -v "$command_name" >/dev/null 2>&1 || {
        echo "$command_name is required" >&2
        exit 1
    }
done
for required_path in "$SDK_SOURCE_ROOT/CMakeLists.txt" "$SDK_SOURCE_ROOT/api/CMakeLists.txt" \
    "$SDK_SOURCE_ROOT/api/src/api/dpu_memory.c" "$SDK_SOURCE_ROOT/api/src/dpu_memory.c"; do
    [[ -f "$required_path" ]] || {
        echo "SDK source path is absent: $required_path" >&2
        exit 1
    }
done

mkdir -p "$RESULT_ROOT"
SDK_COPY="$RESULT_ROOT/sdk_source"
SDK_BUILD="$RESULT_ROOT/sdk_build"
SHADOW_LIB="$RESULT_ROOT/shadow_lib"
mkdir -p "$SDK_COPY" "$SDK_BUILD" "$SHADOW_LIB"
stage "verify SDK source and capture host provenance"

uname -a > "$RESULT_ROOT/uname.txt"
lscpu > "$RESULT_ROOT/lscpu.txt"
numactl --hardware > "$RESULT_ROOT/numa.txt" 2>&1
numactl --show > "$RESULT_ROOT/numactl_show.txt" 2>&1
git -C "$PRIM_ROOT" rev-parse HEAD > "$RESULT_ROOT/prim_git_commit.txt"
git -C "$PRIM_ROOT" status --short > "$RESULT_ROOT/prim_git_status.txt"
dpu-upmem-dpurte-clang --version > "$RESULT_ROOT/dpu_compiler_version.txt" 2>&1
dpu-pkg-config --cflags --libs dpu > "$RESULT_ROOT/dpu_sdk_flags.txt" 2>&1
dpkg-query -W -f='${Package}\t${Version}\t${Status}\n' upmem \
    > "$RESULT_ROOT/upmem_package.txt" 2>&1
INSTALLED_SDK_VERSION="$(dpkg-query -W -f='${Version}' upmem)"
case "$INSTALLED_SDK_VERSION" in
    2025.1.0*) ;;
    *)
        echo "installed UPMEM package version differs: $INSTALLED_SDK_VERSION" >&2
        exit 1
        ;;
esac

(
    cd "$SDK_SOURCE_ROOT"
    sha256sum -c "$SDK_HASH_MANIFEST"
) > "$RESULT_ROOT/sdk_source_baseline_check.txt" 2>&1
cp -a "$SDK_SOURCE_ROOT/." "$SDK_COPY/"
(
    cd "$SDK_COPY"
    patch --dry-run -p1 < "$SDK_PATCH"
    patch -p1 < "$SDK_PATCH"
    printf 'PATCH_APPLIED=PASS\n'
) > "$RESULT_ROOT/sdk_patch_check.txt" 2>&1
(
    cd "$SDK_COPY"
    find . -type f -print0 | sort -z | xargs -0 sha256sum
) > "$RESULT_ROOT/sdk_source.sha256"

stage "build instrumented libdpu"
{
    cmake -S "$SDK_COPY" -B "$SDK_BUILD" \
        -DUPMEM_VERSION="$SDK_VERSION" \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build "$SDK_BUILD" --target dpu --parallel "$BUILD_JOBS"
} > "$RESULT_ROOT/sdk_build.log" 2>&1

INSTRUMENTED_LIB="$(find "$SDK_BUILD" -type f -name 'libdpu.so.2025.1' -print -quit)"
if [[ -z "$INSTRUMENTED_LIB" ]]; then
    echo "instrumented libdpu.so.2025.1 is absent after build" >&2
    exit 1
fi
SYSTEM_LIBDPU="$(ldconfig -p | awk '$1 == "libdpu.so.2025.1" { print $NF; exit }')"
if [[ -z "$SYSTEM_LIBDPU" || ! -f "$SYSTEM_LIBDPU" ]]; then
    echo "system libdpu.so.2025.1 is absent" >&2
    exit 1
fi
SYSTEM_LIB_DIR="$(dirname "$SYSTEM_LIBDPU")"
SYSTEM_LIBDPUHW="$(ldconfig -p | awk '$1 == "libdpuhw.so.2025.1" { print $NF; exit }')"
if [[ -z "$SYSTEM_LIBDPUHW" || ! -f "$SYSTEM_LIBDPUHW" ]]; then
    echo "system libdpuhw.so.2025.1 is absent" >&2
    exit 1
fi
SYSTEM_UPMEM_SHARE="$(dpkg-query -L upmem | awk '
    !found && /\/share\/upmem\/include\/misc(\/|$)/ {
        sub(/\/include\/misc.*/, "")
        print
        found = 1
    }
')"
if [[ -z "$SYSTEM_UPMEM_SHARE" || ! -d "$SYSTEM_UPMEM_SHARE/include/misc" ]]; then
    echo "system UPMEM runtime assets are absent" >&2
    exit 1
fi
mapfile -t SYSTEM_PREDEFINED_PROGRAMS < <(
    find "$SYSTEM_UPMEM_SHARE/include/misc" -maxdepth 1 \
        \( -type f -o -type l \) -print | sort
)
if (( ${#SYSTEM_PREDEFINED_PROGRAMS[@]} == 0 )); then
    echo "system UPMEM predefined programs are absent" >&2
    exit 1
fi
cp -L "$INSTRUMENTED_LIB" "$SHADOW_LIB/libdpu.so.2025.1"
ln -s libdpu.so.2025.1 "$SHADOW_LIB/libdpu.so"
while read -r library_name system_library; do
    case "$library_name" in
        libdpu.so|libdpu.so.2025.1) continue ;;
    esac
    [[ -e "$SHADOW_LIB/$library_name" ]] || ln -s "$system_library" "$SHADOW_LIB/$library_name"
done < <(ldconfig -p | awk '$1 ~ /^libdpu.*\.so/ { print $1, $NF }' | sort -u)
[[ -e "$SHADOW_LIB/libdpuhw.so" ]] || ln -s "$SYSTEM_LIBDPUHW" "$SHADOW_LIB/libdpuhw.so"
[[ -e "$SHADOW_LIB/libdpuhw.so.2025.1" ]] || ln -s "$SYSTEM_LIBDPUHW" "$SHADOW_LIB/libdpuhw.so.2025.1"
if [[ ! -e "$SHADOW_LIB/libdpuhw.so" || ! -e "$SHADOW_LIB/libdpuhw.so.2025.1" ]]; then
    echo "shadow library lacks the system hardware backend" >&2
    exit 1
fi
mkdir -p "$RESULT_ROOT/share"
ln -s "$SYSTEM_UPMEM_SHARE" "$RESULT_ROOT/share/upmem"
sha256sum "$SHADOW_LIB/libdpu.so.2025.1" > "$RESULT_ROOT/instrumented_library.sha256"
{
    printf 'libdpuhw.so=%s\n' "$(readlink -f "$SHADOW_LIB/libdpuhw.so")"
    printf 'libdpuhw.so.2025.1=%s\n' "$(readlink -f "$SHADOW_LIB/libdpuhw.so.2025.1")"
    sha256sum "$SYSTEM_LIBDPUHW"
} > "$RESULT_ROOT/system_backend_libraries.txt"
{
    printf 'share/upmem=%s\n' "$(readlink -f "$RESULT_ROOT/share/upmem")"
    sha256sum "${SYSTEM_PREDEFINED_PROGRAMS[@]}"
} > "$RESULT_ROOT/system_runtime_assets.txt"

cd "$SCRIPT_DIR"
stage "build VA host and DPU binaries"
{
    make clean
    make NR_DPUS=1 NR_TASKLETS="$TASKLETS" BL="$BLOCK_SIZE_LOG2" \
        TYPE=INT32 ENERGY=0 VA_VALIDATION_INPUT=1 all
} > "$RESULT_ROOT/app_build.log" 2>&1
sha256sum bin/host_code bin/dpu_code > "$RESULT_ROOT/binaries.sha256"
sha256sum Makefile host/app.c dpu/task.c support/common.h support/params.h \
    validate_va_baseline_trace.py validate_va_transfer_breakdown.py \
    validate_va_transfer_sweep.py test_validate_va_baseline_trace.py \
    test_validate_va_transfer_breakdown.py \
    test_validate_va_transfer_sweep.py run_hw_transfer_breakdown.sh \
    run_hw_transfer_size_sweep.sh TRANSFER_BREAKDOWN.md \
    sdk/upmem-2025.1.0-source.sha256 \
    sdk/upmem-2025.1.0-transfer-trace.patch sdk/README.md \
    > "$RESULT_ROOT/source.sha256"

export VA_SOURCE_SHA256
export VA_HOST_BINARY_SHA256
export VA_DPU_BINARY_SHA256
export VA_SDK_VERSION
export VA_HOST_NUMA_NODE
VA_SOURCE_SHA256="$(sha256sum "$RESULT_ROOT/source.sha256" | awk '{print $1}')"
VA_HOST_BINARY_SHA256="$(sha256sum bin/host_code | awk '{print $1}')"
VA_DPU_BINARY_SHA256="$(sha256sum bin/dpu_code | awk '{print $1}')"
VA_SDK_VERSION="$SDK_VERSION"
VA_HOST_NUMA_NODE="$NUMA_NODE"

printf '%s\n' \
    "VA_TRANSFER_COLLECTION_MODE=$COLLECTION_MODE" \
    "INPUT_ELEMENTS=$INPUT_ELEMENTS" \
    "VA_SWEEP_INPUT_ELEMENTS=$SWEEP_INPUT_ELEMENTS_TEXT" \
    "VA_SWEEP_OVERHEAD_ELEMENTS=$SWEEP_OVERHEAD_ELEMENTS_TEXT" \
    "TASKLETS=$TASKLETS" \
    "BLOCK_SIZE_LOG2=$BLOCK_SIZE_LOG2" \
    "SCALING=strong" \
    "N_WARMUP_PROCESSES=$N_WARMUP_PROCESSES" \
    "N_REPS_PROCESSES=$N_REPS_PROCESSES" \
    "N_OVERHEAD_PROCESSES=$N_OVERHEAD_PROCESSES" \
    "N_WARMUP_IN_PROCESS=0" \
    "N_REPS_IN_PROCESS=1" \
    "NUMA_NODE=$NUMA_NODE" \
    "VA_DPU_PROFILE=$DPU_PROFILE" \
    "VA_VALIDATION_INPUT=1" \
    "SDK_VERSION=$SDK_VERSION" \
    "SDK_SOURCE_ROOT=$SDK_SOURCE_ROOT" \
    > "$RESULT_ROOT/config.txt"

{
    printf '[readelf host]\n'
    readelf -d bin/host_code
    printf '\n[ldd instrumented]\n'
    LD_LIBRARY_PATH="$SHADOW_LIB:$SYSTEM_LIB_DIR" ldd bin/host_code
    printf '\n[ldd system hardware backend]\n'
    LD_LIBRARY_PATH="$SHADOW_LIB:$SYSTEM_LIB_DIR" ldd "$SHADOW_LIB/libdpuhw.so"
    printf '\n[shadow directory]\n'
    find "$SHADOW_LIB" -maxdepth 1 -printf '%p -> %l\n' | sort
    printf '\n[runtime assets]\n'
    find "$RESULT_ROOT/share" -maxdepth 1 -printf '%p -> %l\n' | sort
} > "$RESULT_ROOT/dynamic_library_resolution.txt" 2>&1

require_correct_result() {
    local log_path="$1"
    grep -q '\[OK\] Outputs are equal' "$log_path"
    grep -q 'VA_CHECKSUM expected=' "$log_path"
}

run_va() {
    local library_mode="$1"
    local allocation_mode="$2"
    local input_elements="$3"
    local transfer_order="$4"
    local run_id="$5"
    local app_trace="$6"
    local baseline_trace="$7"
    local sdk_trace="$8"
    local perf_trace="$9"
    local log_path="${10}"
    local library_path
    local run_rc=0
    local -a command_line

    if [[ "$library_mode" == "instrumented" ]]; then
        library_path="$SHADOW_LIB:$SYSTEM_LIB_DIR"
    else
        library_path="$SYSTEM_LIB_DIR"
    fi
    command_line=(
        numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE"
        ./bin/host_code -w 0 -e 1 -i "$input_elements" -x 1
    )
    (
        export LD_LIBRARY_PATH="$library_path"
        export UPMEM_RUNTIME_LIBRARY_PATH="$SHADOW_LIB"
        export VA_ALLOCATION_MODE="$allocation_mode"
        export VA_DPU_PROFILE="$DPU_PROFILE"
        export VA_TRANSFER_ORDER="$transfer_order"
        export VA_TRACE_RUN_ID="$run_id"
        export VA_TRACE_REPEAT_ID=1
        if [[ "$app_trace" == "-" ]]; then
            unset VA_TRANSFER_BREAKDOWN_CSV
        else
            export VA_TRANSFER_BREAKDOWN_CSV="$app_trace"
        fi
        if [[ "$baseline_trace" == "-" ]]; then
            unset VA_TRACE_CSV
        else
            export VA_TRACE_CSV="$baseline_trace"
        fi
        if [[ "$sdk_trace" == "-" ]]; then
            unset UPMEM_TRANSFER_TRACE_CSV UPMEM_TRANSFER_TRACE_RUN_ID
        else
            export UPMEM_TRANSFER_TRACE_CSV="$sdk_trace"
            export UPMEM_TRANSFER_TRACE_RUN_ID="$run_id"
        fi
        if [[ "$perf_trace" == "-" ]]; then
            "${command_line[@]}"
        else
            perf stat -x, -o "$perf_trace" -e task-clock,cycles,instructions -- \
                "${command_line[@]}"
        fi
    ) > "$log_path" 2>&1 || run_rc=$?
    if (( run_rc != 0 )); then
        printf 'VA run failed: run_id=%s rc=%s log=%s\n' "$run_id" "$run_rc" "$log_path" >&2
        tail -n 80 "$log_path" >&2
        return "$run_rc"
    fi
    if ! require_correct_result "$log_path"; then
        printf 'VA result check failed: run_id=%s log=%s\n' "$run_id" "$log_path" >&2
        tail -n 80 "$log_path" >&2
        return 1
    fi
}

mkdir -p "$RESULT_ROOT/smoke"
stage "run single-DPU and full-rank smoke checks"
run_va instrumented single "$INPUT_ELEMENTS" AB smoke_single \
    "$RESULT_ROOT/smoke/single_app.csv" "$RESULT_ROOT/smoke/single_baseline.csv" \
    "$RESULT_ROOT/smoke/single_sdk.csv" - "$RESULT_ROOT/smoke/single.log"
run_va instrumented rank "$INPUT_ELEMENTS" AB smoke_rank \
    "$RESULT_ROOT/smoke/rank_app.csv" "$RESULT_ROOT/smoke/rank_baseline.csv" \
    "$RESULT_ROOT/smoke/rank_sdk.csv" - "$RESULT_ROOT/smoke/rank.log"
python3 "$SCRIPT_DIR/validate_va_baseline_trace.py" \
    --expected-reps 1 \
    --input-elements "$INPUT_ELEMENTS" \
    --tasklets "$TASKLETS" \
    --block-size-log2 "$BLOCK_SIZE_LOG2" \
    --provenance-root "$RESULT_ROOT" \
    --output-csv "$RESULT_ROOT/smoke/va_hw_events.csv" \
    --manifest "$RESULT_ROOT/smoke/va_hardware_manifest.json" \
    "$RESULT_ROOT/smoke/single_baseline.csv" "$RESULT_ROOT/smoke/rank_baseline.csv" \
    > "$RESULT_ROOT/smoke/validation.log" 2>&1

order_for_rep() {
    if (( $1 % 2 == 1 )); then
        printf 'AB\n'
    else
        printf 'BA\n'
    fi
}

if [[ "$COLLECTION_MODE" == "anchor" ]]; then
    mkdir -p "$RESULT_ROOT/overhead/stock" "$RESULT_ROOT/overhead/instrumented"
    stage "collect stock and instrumented overhead samples"
    for rep in $(seq 1 "$N_OVERHEAD_PROCESSES"); do
        rep_id="$(printf '%02d' "$rep")"
        run_va stock rank "$INPUT_ELEMENTS" AB "overhead_stock_$rep_id" \
            "$RESULT_ROOT/overhead/stock/app_$rep_id.csv" - - - \
            "$RESULT_ROOT/overhead/stock/run_$rep_id.log"
        run_va instrumented rank "$INPUT_ELEMENTS" AB "overhead_instrumented_$rep_id" \
            "$RESULT_ROOT/overhead/instrumented/app_$rep_id.csv" - \
            "$RESULT_ROOT/overhead/instrumented/sdk_$rep_id.csv" - \
            "$RESULT_ROOT/overhead/instrumented/run_$rep_id.log"
    done

    mkdir -p "$RESULT_ROOT/warmup"
    stage "run warmup processes"
    for rep in $(seq 1 "$N_WARMUP_PROCESSES"); do
        rep_id="$(printf '%02d' "$rep")"
        run_va instrumented rank "$INPUT_ELEMENTS" AB "warmup_$rep_id" - - - - \
            "$RESULT_ROOT/warmup/run_$rep_id.log"
    done

    mkdir -p "$RESULT_ROOT/formal"
    stage "collect formal full-rank samples"
    for rep in $(seq 1 "$N_REPS_PROCESSES"); do
        rep_id="$(printf '%02d' "$rep")"
        run_va instrumented rank "$INPUT_ELEMENTS" AB "formal_$rep_id" \
            "$RESULT_ROOT/formal/app_$rep_id.csv" \
            "$RESULT_ROOT/formal/baseline_$rep_id.csv" \
            "$RESULT_ROOT/formal/sdk_$rep_id.csv" \
            "$RESULT_ROOT/formal/perf_$rep_id.csv" \
            "$RESULT_ROOT/formal/run_$rep_id.log"
    done
else
    printf '%s\n' \
        'kind,input_elements,bytes_per_dpu,transfer_order,replicate,library_mode,run_id' \
        > "$RESULT_ROOT/collection_plan.csv"
    for input_elements in "${SWEEP_INPUT_ELEMENTS[@]}"; do
        bytes_per_dpu=$((input_elements / 64 * 4))
        for rep in $(seq 1 "$N_WARMUP_PROCESSES"); do
            rep_id="$(printf '%02d' "$rep")"
            order="$(order_for_rep "$rep")"
            printf 'warmup,%s,%s,%s,%s,instrumented,warmup_n%s_%s_%s\n' \
                "$input_elements" "$bytes_per_dpu" "$order" "$rep" \
                "$input_elements" "${order,,}" "$rep_id" \
                >> "$RESULT_ROOT/collection_plan.csv"
        done
        for rep in $(seq 1 "$N_REPS_PROCESSES"); do
            rep_id="$(printf '%02d' "$rep")"
            order="$(order_for_rep "$rep")"
            printf 'formal,%s,%s,%s,%s,instrumented,formal_n%s_%s_%s\n' \
                "$input_elements" "$bytes_per_dpu" "$order" "$rep" \
                "$input_elements" "${order,,}" "$rep_id" \
                >> "$RESULT_ROOT/collection_plan.csv"
        done
    done
    for input_elements in "${SWEEP_OVERHEAD_ELEMENTS[@]}"; do
        bytes_per_dpu=$((input_elements / 64 * 4))
        for library_mode in stock instrumented; do
            for rep in $(seq 1 "$N_OVERHEAD_PROCESSES"); do
                rep_id="$(printf '%02d' "$rep")"
                order="$(order_for_rep "$rep")"
                printf 'overhead,%s,%s,%s,%s,%s,overhead_%s_n%s_%s_%s\n' \
                    "$input_elements" "$bytes_per_dpu" "$order" "$rep" \
                    "$library_mode" "$library_mode" "$input_elements" \
                    "${order,,}" "$rep_id" >> "$RESULT_ROOT/collection_plan.csv"
            done
        done
    done

    stage "collect stock and instrumented overhead sweep samples"
    for input_elements in "${SWEEP_OVERHEAD_ELEMENTS[@]}"; do
        mkdir -p "$RESULT_ROOT/overhead/n$input_elements/stock" \
            "$RESULT_ROOT/overhead/n$input_elements/instrumented"
        for rep in $(seq 1 "$N_OVERHEAD_PROCESSES"); do
            rep_id="$(printf '%02d' "$rep")"
            order="$(order_for_rep "$rep")"
            order_slug="${order,,}"
            run_id="overhead_stock_n${input_elements}_${order_slug}_${rep_id}"
            run_va stock rank "$input_elements" "$order" "$run_id" \
                "$RESULT_ROOT/overhead/n$input_elements/stock/app_${order_slug}_$rep_id.csv" \
                - - - \
                "$RESULT_ROOT/overhead/n$input_elements/stock/run_${order_slug}_$rep_id.log"
            run_id="overhead_instrumented_n${input_elements}_${order_slug}_${rep_id}"
            run_va instrumented rank "$input_elements" "$order" "$run_id" \
                "$RESULT_ROOT/overhead/n$input_elements/instrumented/app_${order_slug}_$rep_id.csv" \
                - "$RESULT_ROOT/overhead/n$input_elements/instrumented/sdk_${order_slug}_$rep_id.csv" - \
                "$RESULT_ROOT/overhead/n$input_elements/instrumented/run_${order_slug}_$rep_id.log"
        done
    done

    stage "run size-sweep warmup processes"
    for input_elements in "${SWEEP_INPUT_ELEMENTS[@]}"; do
        for rep in $(seq 1 "$N_WARMUP_PROCESSES"); do
            rep_id="$(printf '%02d' "$rep")"
            order="$(order_for_rep "$rep")"
            order_slug="${order,,}"
            run_dir="$RESULT_ROOT/warmup/n$input_elements/$order_slug"
            mkdir -p "$run_dir"
            run_va instrumented rank "$input_elements" "$order" \
                "warmup_n${input_elements}_${order_slug}_${rep_id}" - - - - \
                "$run_dir/run_$rep_id.log"
        done
    done

    stage "collect formal full-rank size sweep"
    for input_elements in "${SWEEP_INPUT_ELEMENTS[@]}"; do
        stage "collect formal input_elements=$input_elements"
        for rep in $(seq 1 "$N_REPS_PROCESSES"); do
            rep_id="$(printf '%02d' "$rep")"
            order="$(order_for_rep "$rep")"
            order_slug="${order,,}"
            run_dir="$RESULT_ROOT/formal/n$input_elements/$order_slug"
            run_id="formal_n${input_elements}_${order_slug}_${rep_id}"
            mkdir -p "$run_dir"
            run_va instrumented rank "$input_elements" "$order" "$run_id" \
                "$run_dir/app_$rep_id.csv" "$run_dir/baseline_$rep_id.csv" \
                "$run_dir/sdk_$rep_id.csv" "$run_dir/perf_$rep_id.csv" \
                "$run_dir/run_$rep_id.log"
        done
    done
fi

mkdir -p "$RESULT_ROOT/imc"
stage "collect IMC status"
if perf list 2>/dev/null | grep -q 'uncore_imc.*cas_count_read'; then
    set +e
    (
        export LD_LIBRARY_PATH="$SHADOW_LIB:$SYSTEM_LIB_DIR"
        export UPMEM_RUNTIME_LIBRARY_PATH="$SHADOW_LIB"
        export VA_ALLOCATION_MODE=rank
        export VA_DPU_PROFILE="$DPU_PROFILE"
        export VA_TRANSFER_ORDER=AB
        perf stat -a -x, -o "$RESULT_ROOT/imc/perf.csv" \
            -e 'uncore_imc_*/cas_count_read/' -e 'uncore_imc_*/cas_count_write/' -- \
            numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE" \
            ./bin/host_code -w 0 -e 1 -i "$INPUT_ELEMENTS" -x 1
    ) > "$RESULT_ROOT/imc/run.log" 2>&1
    imc_rc=$?
    set -e
    if [[ "$imc_rc" == "0" ]]; then
        require_correct_result "$RESULT_ROOT/imc/run.log"
        printf 'COLLECTED\n' > "$RESULT_ROOT/imc_status.txt"
    elif grep -Eqi 'permission|access|paranoid|operation not permitted' \
        "$RESULT_ROOT/imc/run.log" "$RESULT_ROOT/imc/perf.csv" 2>/dev/null; then
        printf 'PERMISSION_DENIED rc=%s\n' "$imc_rc" > "$RESULT_ROOT/imc_status.txt"
    else
        printf 'COLLECTION_ERROR rc=%s\n' "$imc_rc" > "$RESULT_ROOT/imc_status.txt"
    fi
else
    printf 'EVENT_UNAVAILABLE\n' > "$RESULT_ROOT/imc_status.txt"
fi

set +e
stage "validate traces and create archive"
if [[ "$COLLECTION_MODE" == "anchor" ]]; then
    mapfile -t formal_app < <(find "$RESULT_ROOT/formal" -type f -name 'app_*.csv' | sort)
    mapfile -t formal_sdk < <(find "$RESULT_ROOT/formal" -type f -name 'sdk_*.csv' | sort)
    mapfile -t formal_baseline < <(find "$RESULT_ROOT/formal" -type f -name 'baseline_*.csv' | sort)
    mapfile -t formal_perf < <(find "$RESULT_ROOT/formal" -type f -name 'perf_*.csv' | sort)
    mapfile -t stock_app < <(find "$RESULT_ROOT/overhead/stock" -type f -name 'app_*.csv' | sort)
    mapfile -t instrumented_app < <(find "$RESULT_ROOT/overhead/instrumented" -type f -name 'app_*.csv' | sort)
    python3 "$SCRIPT_DIR/validate_va_transfer_breakdown.py" \
        --app-trace "${formal_app[@]}" \
        --sdk-trace "${formal_sdk[@]}" \
        --baseline-trace "${formal_baseline[@]}" \
        --stock-app-trace "${stock_app[@]}" \
        --instrumented-app-trace "${instrumented_app[@]}" \
        --perf "${formal_perf[@]}" \
        --expected-reps "$N_REPS_PROCESSES" \
        --input-elements "$INPUT_ELEMENTS" \
        --tasklets "$TASKLETS" \
        --block-size-log2 "$BLOCK_SIZE_LOG2" \
        --provenance-root "$RESULT_ROOT" \
        --output-dir "$RESULT_ROOT/summary" \
        > "$RESULT_ROOT/validation.log" 2>&1
else
    python3 "$SCRIPT_DIR/validate_va_transfer_sweep.py" \
        --result-root "$RESULT_ROOT" \
        --input-elements "${SWEEP_INPUT_ELEMENTS[@]}" \
        --overhead-input-elements "${SWEEP_OVERHEAD_ELEMENTS[@]}" \
        --expected-reps-per-size "$N_REPS_PROCESSES" \
        --expected-warmups-per-size "$N_WARMUP_PROCESSES" \
        --expected-overhead-reps "$N_OVERHEAD_PROCESSES" \
        --tasklets "$TASKLETS" \
        --block-size-log2 "$BLOCK_SIZE_LOG2" \
        --output-dir "$RESULT_ROOT/summary" \
        > "$RESULT_ROOT/validation.log" 2>&1
fi
validation_rc=$?
set -e

ARCHIVE="${RESULT_ROOT}.tar.gz"
tar -czf "$ARCHIVE" -C "$(dirname "$RESULT_ROOT")" "$(basename "$RESULT_ROOT")"
(
    cd "$(dirname "$ARCHIVE")"
    sha256sum "$(basename "$ARCHIVE")"
) > "${ARCHIVE}.sha256"

if [[ "$validation_rc" != "0" ]]; then
    echo "FAIL VA transfer $COLLECTION_MODE validation"
    echo "Result: $RESULT_ROOT"
    echo "Archive: $ARCHIVE"
    exit "$validation_rc"
fi
echo "PASS VA transfer $COLLECTION_MODE collection"
echo "Result: $RESULT_ROOT"
echo "Archive: $ARCHIVE"
