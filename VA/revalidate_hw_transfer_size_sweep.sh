#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULT_ROOT="${1:-${RESULT_ROOT:-}}"

if [[ -z "$RESULT_ROOT" || ! -d "$RESULT_ROOT" ]]; then
    echo "usage: $0 RESULT_ROOT" >&2
    exit 1
fi
if [[ ! -f "$RESULT_ROOT/config.txt" || ! -f "$RESULT_ROOT/collection_plan.csv" ]]; then
    echo "result root lacks the size-sweep configuration: $RESULT_ROOT" >&2
    exit 1
fi

config_value() {
    sed -n "s/^$1=//p" "$RESULT_ROOT/config.txt" | tail -n 1
}

INPUT_ELEMENTS_TEXT="$(config_value VA_SWEEP_INPUT_ELEMENTS)"
OVERHEAD_INPUT_ELEMENTS_TEXT="$(config_value VA_SWEEP_OVERHEAD_ELEMENTS)"
EXPECTED_REPS="$(config_value N_REPS_PROCESSES)"
EXPECTED_WARMUPS="$(config_value N_WARMUP_PROCESSES)"
EXPECTED_OVERHEAD_REPS="$(config_value N_OVERHEAD_PROCESSES)"
TASKLETS="$(config_value TASKLETS)"
BLOCK_SIZE_LOG2="$(config_value BLOCK_SIZE_LOG2)"
STRICT_OVERHEAD="${VA_SWEEP_STRICT_OVERHEAD:-0}"

if [[ -z "$INPUT_ELEMENTS_TEXT" || -z "$OVERHEAD_INPUT_ELEMENTS_TEXT" ||
      -z "$EXPECTED_REPS" || -z "$EXPECTED_WARMUPS" ||
      -z "$EXPECTED_OVERHEAD_REPS" || -z "$TASKLETS" ||
      -z "$BLOCK_SIZE_LOG2" ]]; then
    echo "size-sweep configuration is incomplete: $RESULT_ROOT/config.txt" >&2
    exit 1
fi
if [[ "$STRICT_OVERHEAD" != "0" && "$STRICT_OVERHEAD" != "1" ]]; then
    echo "VA_SWEEP_STRICT_OVERHEAD must be 0 or 1: $STRICT_OVERHEAD" >&2
    exit 1
fi

read -r -a INPUT_ELEMENTS <<< "$INPUT_ELEMENTS_TEXT"
read -r -a OVERHEAD_INPUT_ELEMENTS <<< "$OVERHEAD_INPUT_ELEMENTS_TEXT"
OUTPUT_DIR="$RESULT_ROOT/summary_revalidated_v2"
LOG_PATH="$RESULT_ROOT/validation_revalidated_v2.log"
PROVENANCE_PATH="$RESULT_ROOT/revalidation_v2_provenance.txt"
mkdir -p "$OUTPUT_DIR"

{
    printf 'schema_version=upmem.va_transfer_size_sweep_revalidation.v2\n'
    printf 'validator_commit=%s\n' "$(git -C "$PRIM_ROOT" rev-parse HEAD)"
    printf 'validator_git_status_begin\n'
    git -C "$PRIM_ROOT" status --short
    printf 'validator_git_status_end\n'
    sha256sum "$SCRIPT_DIR/validate_va_transfer_sweep.py" \
        "$SCRIPT_DIR/validate_va_transfer_breakdown.py"
} > "$PROVENANCE_PATH"

validation_args=(
    python3 "$SCRIPT_DIR/validate_va_transfer_sweep.py"
    --result-root "$RESULT_ROOT"
    --input-elements "${INPUT_ELEMENTS[@]}"
    --overhead-input-elements "${OVERHEAD_INPUT_ELEMENTS[@]}"
    --expected-reps-per-size "$EXPECTED_REPS"
    --expected-warmups-per-size "$EXPECTED_WARMUPS"
    --expected-overhead-reps "$EXPECTED_OVERHEAD_REPS"
    --tasklets "$TASKLETS"
    --block-size-log2 "$BLOCK_SIZE_LOG2"
    --output-dir "$OUTPUT_DIR"
)
if [[ "$STRICT_OVERHEAD" == "1" ]]; then
    validation_args+=(--strict-overhead)
fi

set +e
"${validation_args[@]}" > "$LOG_PATH" 2>&1
validation_rc=$?
set -e
cat "$LOG_PATH"
if (( validation_rc != 0 )); then
    exit "$validation_rc"
fi

ARCHIVE="${RESULT_ROOT}.revalidated_v2.tar.gz"
tar -czf "$ARCHIVE" -C "$(dirname "$RESULT_ROOT")" "$(basename "$RESULT_ROOT")"
(
    cd "$(dirname "$ARCHIVE")"
    sha256sum "$(basename "$ARCHIVE")"
) > "${ARCHIVE}.sha256"

echo "PASS VA transfer sweep revalidation"
echo "Summary: $OUTPUT_DIR"
echo "Archive: $ARCHIVE"
