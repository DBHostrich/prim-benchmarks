#!/usr/bin/env python3
"""Validate the full-rank VA transfer size sweep."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import validate_va_transfer_breakdown as base


PLAN_HEADER = (
    "kind", "input_elements", "bytes_per_dpu", "transfer_order", "replicate",
    "library_mode", "run_id",
)
SAMPLE_HEADER = (
    "run_id", "input_elements", "bytes_per_dpu", "aggregate_bytes",
    "transfer_order", "component", "execution_position", "direction",
    "c_pre_wall_ns", "t_backend_wall_ns", "c_tail_wall_ns",
    "backend_process_cpu_ns", "backend_share_pct", "cpu_parallelism",
    "effective_bandwidth_GBps",
)
SUMMARY_HEADER = (
    "input_elements", "bytes_per_dpu", "transfer_order", "component",
    "execution_position", "direction", "samples", "c_pre_p50_ns", "c_pre_p90_ns",
    "backend_p50_ns", "backend_p90_ns", "c_tail_p50_ns", "c_tail_p90_ns",
    "backend_process_cpu_p50_ns", "backend_process_cpu_p90_ns",
    "backend_share_p50_pct", "cpu_parallelism_p50", "effective_bandwidth_p50_GBps",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--input-elements", nargs="+", type=int, required=True)
    parser.add_argument("--overhead-input-elements", nargs="+", type=int, required=True)
    parser.add_argument("--expected-reps-per-size", type=int, default=30)
    parser.add_argument("--expected-warmups-per-size", type=int, default=5)
    parser.add_argument("--expected-overhead-reps", type=int, default=10)
    parser.add_argument("--tasklets", type=int, default=16)
    parser.add_argument("--block-size-log2", type=int, default=10)
    parser.add_argument("--p50-threshold-pct", type=float, default=3.0)
    parser.add_argument("--p90-threshold-pct", type=float, default=5.0)
    parser.add_argument(
        "--strict-overhead",
        action="store_true",
        help="return a failure status when an overhead threshold is exceeded",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_plan(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != PLAN_HEADER:
            base.fail(f"{path}: collection plan header differs")
        rows = list(reader)
    if not rows:
        base.fail(f"{path}: collection plan is empty")
    return rows


def expected_bytes_per_dpu(input_elements: int) -> int:
    return base.aligned_per_dpu_elements(input_elements, 64) * 4


def validate_plan(
    rows: list[dict[str, str]],
    input_elements: list[int],
    overhead_input_elements: list[int],
    reps_per_size: int,
    warmups_per_size: int,
    overhead_reps: int,
) -> dict[str, dict[str, str]]:
    if reps_per_size <= 0 or reps_per_size % 2 != 0:
        base.fail("formal repetitions per size must be a positive even number")
    expected_sizes = set(input_elements)
    expected_overhead_sizes = set(overhead_input_elements)
    if len(expected_sizes) != len(input_elements) or len(expected_overhead_sizes) != len(overhead_input_elements):
        base.fail("input size lists contain duplicates")
    if not expected_overhead_sizes <= expected_sizes:
        base.fail("overhead sizes must be part of the formal sweep")

    by_run: dict[str, dict[str, str]] = {}
    counts: Counter[tuple[str, int, str, str]] = Counter()
    replicate_keys: set[tuple[str, int, str, int]] = set()
    for row in rows:
        try:
            size = int(row["input_elements"])
            bytes_per_dpu = int(row["bytes_per_dpu"])
            replicate = int(row["replicate"])
        except ValueError as error:
            base.fail(f"collection plan contains a non-integer field: {error}")
        if row["kind"] not in ("formal", "warmup", "overhead"):
            base.fail(f"collection plan kind differs: {row['kind']}")
        if row["transfer_order"] not in ("AB", "BA"):
            base.fail(f"collection plan transfer order differs: {row['transfer_order']}")
        if row["library_mode"] not in ("instrumented", "stock"):
            base.fail(f"collection plan library mode differs: {row['library_mode']}")
        if size not in expected_sizes or bytes_per_dpu != expected_bytes_per_dpu(size):
            base.fail(f"collection plan size or byte count differs for {row['run_id']}")
        if replicate <= 0 or not row["run_id"]:
            base.fail("collection plan replicate or run ID differs")
        limit = {
            "formal": reps_per_size,
            "warmup": warmups_per_size,
            "overhead": overhead_reps,
        }[row["kind"]]
        if replicate > limit:
            base.fail(f"{row['run_id']}: collection plan replicate exceeds its quota")
        if row["run_id"] in by_run:
            base.fail(f"duplicate collection plan run ID: {row['run_id']}")
        replicate_key = (row["kind"], size, row["library_mode"], replicate)
        if replicate_key in replicate_keys:
            base.fail(f"{row['run_id']}: duplicate collection plan replicate")
        replicate_keys.add(replicate_key)
        if row["kind"] in ("formal", "warmup") and row["library_mode"] != "instrumented":
            base.fail(f"{row['run_id']}: formal and warmup rows require the instrumented library")
        if row["kind"] == "overhead" and size not in expected_overhead_sizes:
            base.fail(f"{row['run_id']}: overhead size differs")
        by_run[row["run_id"]] = row
        counts[(row["kind"], size, row["library_mode"], row["transfer_order"])] += 1

    for size in input_elements:
        for order in ("AB", "BA"):
            if counts[("formal", size, "instrumented", order)] != reps_per_size // 2:
                base.fail(f"size {size}: formal {order} quota differs")
        warmup_total = sum(
            counts[("warmup", size, "instrumented", order)] for order in ("AB", "BA")
        )
        if warmup_total != warmups_per_size:
            base.fail(f"size {size}: warmup quota differs")
    for size in overhead_input_elements:
        for library_mode in ("stock", "instrumented"):
            total = sum(
                counts[("overhead", size, library_mode, order)] for order in ("AB", "BA")
            )
            if total != overhead_reps:
                base.fail(f"size {size}: {library_mode} overhead quota differs")
    return by_run


def validate_row_orders(
    groups: dict[tuple[str, int], dict[str, dict[str, str]]],
    plan_by_run: dict[str, dict[str, str]],
    kind: str,
    library_mode: str,
    input_elements: int | None = None,
) -> None:
    actual_runs = {run_id for run_id, _repeat in groups}
    expected_runs = {
        run_id for run_id, row in plan_by_run.items()
        if row["kind"] == kind and row["library_mode"] == library_mode
        and (input_elements is None or int(row["input_elements"]) == input_elements)
    }
    if actual_runs != expected_runs:
        base.fail(f"{kind} {library_mode} run IDs differ from the collection plan")
    for (run_id, _repeat), indexed in groups.items():
        order = base.detect_transfer_order(indexed)
        if order != plan_by_run[run_id]["transfer_order"]:
            base.fail(f"{run_id}: observed transfer order differs from the collection plan")


def validate_baseline_orders(
    rows: list[dict[str, str]],
    plan_by_run: dict[str, dict[str, str]],
) -> None:
    sequences = {
        "AB": "prepare_args>push_args>prepare_A>push_A>prepare_B>push_B",
        "BA": "prepare_args>push_args>prepare_B>push_B>prepare_A>push_A",
    }
    expected_runs = {
        run_id for run_id, row in plan_by_run.items()
        if row["kind"] == "formal" and row["library_mode"] == "instrumented"
    }
    actual_runs = {row["run_id"] for row in rows if row["phase"] == "H2D"}
    if actual_runs != expected_runs:
        base.fail("baseline H2D run IDs differ from the collection plan")
    for row in rows:
        if row["phase"] != "H2D":
            continue
        plan = plan_by_run[row["run_id"]]
        if row["call_sequence"] != sequences[plan["transfer_order"]]:
            base.fail(f"{row['run_id']}: baseline H2D call sequence differs")


def require_success_logs(root: Path, expected_count: int, pattern: str) -> None:
    paths = sorted(root.glob(pattern))
    if len(paths) != expected_count:
        base.fail(f"expected {expected_count} logs for {pattern}, found {len(paths)}")
    for path in paths:
        text = path.read_text()
        if "[OK] Outputs are equal" not in text or "VA_CHECKSUM expected=" not in text:
            base.fail(f"{path}: functional result differs")


def overhead_requires_failure(overhead_passed: bool, strict_overhead: bool) -> bool:
    return strict_overhead and not overhead_passed


def make_samples(
    derived: list[dict[str, object]],
    plan_by_run: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in derived:
        plan = plan_by_run[str(row["run_id"])]
        component = str(row["component"])
        order = plan["transfer_order"]
        if component == "PUSH_C":
            position = "D2H"
        elif (component == "PUSH_A" and order == "AB") or (component == "PUSH_B" and order == "BA"):
            position = "FIRST"
        else:
            position = "SECOND"
        pre = int(row["c_pre_wall_ns"])
        backend = int(row["t_backend_wall_ns"])
        tail = int(row["c_tail_wall_ns"])
        process_cpu = int(row["backend_process_cpu_ns"])
        push = pre + backend + tail
        aggregate_bytes = int(row["aggregate_bytes"])
        if backend <= 0 or push <= 0:
            base.fail(f"{row['run_id']}: coupled backend duration must be positive")
        output.append({
            "run_id": row["run_id"],
            "input_elements": int(plan["input_elements"]),
            "bytes_per_dpu": int(row["bytes_per_dpu"]),
            "aggregate_bytes": aggregate_bytes,
            "transfer_order": order,
            "component": component,
            "execution_position": position,
            "direction": row["direction"],
            "c_pre_wall_ns": pre,
            "t_backend_wall_ns": backend,
            "c_tail_wall_ns": tail,
            "backend_process_cpu_ns": process_cpu,
            "backend_share_pct": f"{backend * 100.0 / push:.9f}",
            "cpu_parallelism": f"{process_cpu / backend:.9f}",
            "effective_bandwidth_GBps": f"{aggregate_bytes / backend:.9f}",
        })
    return output


def summarize_samples(samples: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, int, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in samples:
        key = (
            int(row["input_elements"]), int(row["bytes_per_dpu"]),
            str(row["transfer_order"]), str(row["component"]),
            str(row["execution_position"]), str(row["direction"]),
        )
        groups[key].append(row)
    output: list[dict[str, object]] = []
    for key in sorted(groups):
        size, bytes_per_dpu, order, component, position, direction = key
        rows = groups[key]
        values = lambda field: [float(row[field]) for row in rows]
        output.append({
            "input_elements": size,
            "bytes_per_dpu": bytes_per_dpu,
            "transfer_order": order,
            "component": component,
            "execution_position": position,
            "direction": direction,
            "samples": len(rows),
            "c_pre_p50_ns": f"{base.percentile(values('c_pre_wall_ns'), 50):.3f}",
            "c_pre_p90_ns": f"{base.percentile(values('c_pre_wall_ns'), 90):.3f}",
            "backend_p50_ns": f"{base.percentile(values('t_backend_wall_ns'), 50):.3f}",
            "backend_p90_ns": f"{base.percentile(values('t_backend_wall_ns'), 90):.3f}",
            "c_tail_p50_ns": f"{base.percentile(values('c_tail_wall_ns'), 50):.3f}",
            "c_tail_p90_ns": f"{base.percentile(values('c_tail_wall_ns'), 90):.3f}",
            "backend_process_cpu_p50_ns": f"{base.percentile(values('backend_process_cpu_ns'), 50):.3f}",
            "backend_process_cpu_p90_ns": f"{base.percentile(values('backend_process_cpu_ns'), 90):.3f}",
            "backend_share_p50_pct": f"{base.percentile(values('backend_share_pct'), 50):.6f}",
            "cpu_parallelism_p50": f"{base.percentile(values('cpu_parallelism'), 50):.6f}",
            "effective_bandwidth_p50_GBps": f"{base.percentile(values('effective_bandwidth_GBps'), 50):.6f}",
        })
    return output


def main() -> int:
    args = parse_args()
    root = args.result_root
    try:
        plan_rows = load_plan(root / "collection_plan.csv")
        plan_by_run = validate_plan(
            plan_rows, args.input_elements, args.overhead_input_elements,
            args.expected_reps_per_size, args.expected_warmups_per_size,
            args.expected_overhead_reps,
        )
        app_paths = sorted((root / "formal").glob("**/app_*.csv"))
        sdk_paths = sorted((root / "formal").glob("**/sdk_*.csv"))
        baseline_paths = sorted((root / "formal").glob("**/baseline_*.csv"))
        perf_paths = sorted((root / "formal").glob("**/perf_*.csv"))
        app_rows = base.load_csv(app_paths, base.APP_HEADER, base.APP_SCHEMA)
        sdk_rows = base.load_csv(sdk_paths, base.SDK_HEADER, base.SDK_SCHEMA)

        all_groups: dict[tuple[str, int], dict[str, dict[str, str]]] = {}
        all_totals: dict[tuple[str, int], dict[str, int]] = {}
        perf_status: dict[str, dict[str, object]] = {}
        for size in args.input_elements:
            size_rows = [row for row in app_rows if base.as_int(row, "input_elements") == size]
            groups, totals = base.validate_app_rows(
                size_rows, size, args.tasklets, args.block_size_log2,
                args.expected_reps_per_size,
            )
            all_groups.update(groups)
            all_totals.update(totals)
            size_perf = [path for path in perf_paths if f"/n{size}/" in str(path)]
            perf = base.validate_perf(size_perf, args.expected_reps_per_size)
            perf_status[str(size)] = {"status": perf["status"], "files": len(perf["files"])}
        validate_row_orders(all_groups, plan_by_run, "formal", "instrumented")
        baseline_rows = base.validate_baseline_crosscheck(baseline_paths, all_groups, all_totals)
        validate_baseline_orders(baseline_rows, plan_by_run)
        derived = base.validate_sdk_rows(sdk_rows, all_groups)

        overhead_rows: list[dict[str, object]] = []
        overhead_passed = True
        for size in args.overhead_input_elements:
            stock_paths = sorted((root / "overhead" / f"n{size}" / "stock").glob("app_*.csv"))
            instrumented_paths = sorted((root / "overhead" / f"n{size}" / "instrumented").glob("app_*.csv"))
            stock_rows = base.load_csv(stock_paths, base.APP_HEADER, base.APP_SCHEMA)
            instrumented_rows = base.load_csv(instrumented_paths, base.APP_HEADER, base.APP_SCHEMA)
            stock_groups, stock_totals = base.validate_app_rows(
                stock_rows, size, args.tasklets, args.block_size_log2,
                args.expected_overhead_reps,
            )
            instrumented_groups, instrumented_totals = base.validate_app_rows(
                instrumented_rows, size, args.tasklets, args.block_size_log2,
                args.expected_overhead_reps,
            )
            validate_row_orders(stock_groups, plan_by_run, "overhead", "stock", size)
            validate_row_orders(
                instrumented_groups, plan_by_run, "overhead", "instrumented", size
            )
            comparisons, passed = base.compare_overhead(
                stock_totals, instrumented_totals,
                args.p50_threshold_pct, args.p90_threshold_pct,
            )
            for row in comparisons:
                overhead_rows.append({"input_elements": size, **row})
            overhead_passed = overhead_passed and passed

        require_success_logs(
            root, len(args.input_elements) * args.expected_reps_per_size,
            "formal/**/run_*.log",
        )
        require_success_logs(
            root, len(args.input_elements) * args.expected_warmups_per_size,
            "warmup/**/run_*.log",
        )
        require_success_logs(
            root,
            len(args.overhead_input_elements) * args.expected_overhead_reps * 2,
            "overhead/**/run_*.log",
        )
        provenance = base.validate_provenance(root)
        config = (root / "config.txt").read_text()
        if "VA_TRANSFER_COLLECTION_MODE=sweep" not in config:
            base.fail("config does not select the size sweep")
        if "regionMode=perf" not in config:
            base.fail("config does not select regionMode=perf")
        samples = make_samples(derived, plan_by_run)
        summaries = summarize_samples(samples)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        base.write_csv(args.output_dir / "coupled_backend_samples.csv", SAMPLE_HEADER, samples)
        base.write_csv(args.output_dir / "coupled_backend_summary.csv", SUMMARY_HEADER, summaries)
        base.write_csv(
            args.output_dir / "overhead_comparison.csv",
            ("input_elements", "phase", "percentile", "stock_ns", "instrumented_ns",
             "absolute_shift_pct", "threshold_pct", "status"),
            overhead_rows,
        )
        manifest = {
            "schema_version": "upmem.va_transfer_size_sweep_manifest.v2",
            "collection_status": "PASS",
            "functional_status": "PASS",
            "timing_status": "REPORTED" if overhead_passed else "FAIL",
            "overhead_threshold_policy": "STRICT" if args.strict_overhead else "REPORT",
            "overhead_failures": sum(row["status"] == "FAIL" for row in overhead_rows),
            "validator_sha256": base.sha256_file(Path(__file__)),
            "base_validator_sha256": base.sha256_file(Path(base.__file__)),
            "input_elements": args.input_elements,
            "bytes_per_dpu": [expected_bytes_per_dpu(size) for size in args.input_elements],
            "formal_processes": len(all_groups),
            "formal_processes_per_size": args.expected_reps_per_size,
            "transfer_order_counts": Counter(
                plan_by_run[run_id]["transfer_order"] for run_id, _repeat in all_groups
            ),
            "app_rows": len(app_rows),
            "sdk_rows": len(sdk_rows),
            "sdk_matched_mram_pushes": len(derived),
            "coupled_backend_samples": len(samples),
            "overhead": overhead_rows,
            "perf": perf_status,
            "provenance": provenance,
            "collection_plan_sha256": base.sha256_file(root / "collection_plan.csv"),
        }
        manifest["transfer_order_counts"] = dict(manifest["transfer_order_counts"])
        (args.output_dir / "va_transfer_size_sweep_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        if overhead_requires_failure(overhead_passed, args.strict_overhead):
            print(
                "FAIL VA_TRANSFER_SIZE_SWEEP collection_status=PASS "
                "functional_status=PASS timing_status=FAIL strict_overhead=1"
            )
            return 1
    except (OSError, ValueError, base.ValidationError) as error:
        print(f"FAIL VA_TRANSFER_SIZE_SWEEP {error}")
        return 1
    timing_status = "REPORTED" if overhead_passed else "FAIL"
    overhead_failures = sum(row["status"] == "FAIL" for row in overhead_rows)
    print(
        f"PASS VA_TRANSFER_SIZE_SWEEP collection_status=PASS functional_status=PASS "
        f"timing_status={timing_status} overhead_failures={overhead_failures} "
        f"sizes={len(args.input_elements)} processes={len(all_groups)} output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
