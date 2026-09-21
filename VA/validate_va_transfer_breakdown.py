#!/usr/bin/env python3
"""Validate VA application and UPMEM SDK transfer timing traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


APP_SCHEMA = "upmem.va_transfer_breakdown.hw.v1"
SDK_SCHEMA = "upmem.sdk_transfer_trace.v1"
APP_HEADER = (
    "schema_version", "run_id", "repeat_id", "configuration", "actual_dpus",
    "actual_ranks", "tasklets", "block_size_log2", "input_elements", "component",
    "direction", "target_space", "operation", "wall_start_ns", "wall_end_ns",
    "wall_duration_ns", "thread_cpu_start_ns", "thread_cpu_end_ns",
    "thread_cpu_duration_ns", "process_cpu_start_ns", "process_cpu_end_ns",
    "process_cpu_duration_ns", "cpu_start", "cpu_end", "thread_nvcsw_delta",
    "thread_nivcsw_delta", "process_nvcsw_delta", "process_nivcsw_delta",
    "bytes_per_dpu", "aggregate_bytes", "result_ok", "source_sha256",
    "host_binary_sha256", "dpu_binary_sha256",
)
SDK_HEADER = (
    "schema_version", "run_id", "sequence", "pid", "tid", "event_kind", "direction",
    "memory_type", "bytes_per_dpu", "aggregate_bytes", "rank_id", "rank_count",
    "dpu_count", "wall_start_ns", "wall_end_ns", "wall_duration_ns",
    "thread_cpu_start_ns", "thread_cpu_end_ns", "thread_cpu_duration_ns",
    "process_cpu_start_ns", "process_cpu_end_ns", "process_cpu_duration_ns",
    "cpu_start", "cpu_end", "thread_nvcsw_delta", "thread_nivcsw_delta",
    "process_nvcsw_delta", "process_nivcsw_delta", "xfer_flags", "return_status",
    "diagnostic_flags", "overflow_count", "runtime_version", "trace_clock",
)
BASELINE_HEADER = (
    "schema_version", "run_id", "repeat_id", "configuration", "actual_dpus",
    "actual_ranks", "tasklets", "block_size_log2", "input_elements", "scaling",
    "phase", "duration_ns", "bytes_per_dpu", "aggregate_bytes", "call_sequence",
    "result_ok", "expected_checksum", "actual_checksum", "host_numa_node",
    "sdk_version", "source_sha256", "host_binary_sha256", "dpu_binary_sha256",
)
COMPONENTS = (
    "PREPARE_ARGS", "PUSH_ARGS", "PREPARE_A", "PUSH_A", "PREPARE_B", "PUSH_B",
    "PREPARE_C", "PUSH_C",
)
H2D_COMPONENTS = COMPONENTS[:6]
D2H_COMPONENTS = COMPONENTS[6:]
COMPONENT_META = {
    "PREPARE_ARGS": ("H2D", "WRAM", "PREPARE"),
    "PUSH_ARGS": ("H2D", "WRAM", "PUSH"),
    "PREPARE_A": ("H2D", "MRAM", "PREPARE"),
    "PUSH_A": ("H2D", "MRAM", "PUSH"),
    "PREPARE_B": ("H2D", "MRAM", "PREPARE"),
    "PUSH_B": ("H2D", "MRAM", "PUSH"),
    "PREPARE_C": ("D2H", "MRAM", "PREPARE"),
    "PUSH_C": ("D2H", "MRAM", "PUSH"),
}
INTEGER_FIELDS_APP = (
    "repeat_id", "actual_dpus", "actual_ranks", "tasklets", "block_size_log2",
    "input_elements", "wall_start_ns", "wall_end_ns", "wall_duration_ns",
    "thread_cpu_start_ns", "thread_cpu_end_ns", "thread_cpu_duration_ns",
    "process_cpu_start_ns", "process_cpu_end_ns", "process_cpu_duration_ns",
    "cpu_start", "cpu_end", "thread_nvcsw_delta", "thread_nivcsw_delta",
    "process_nvcsw_delta", "process_nivcsw_delta", "bytes_per_dpu",
    "aggregate_bytes", "result_ok",
)
INTEGER_FIELDS_SDK = (
    "sequence", "pid", "tid", "bytes_per_dpu", "aggregate_bytes", "rank_id",
    "rank_count", "dpu_count", "wall_start_ns", "wall_end_ns", "wall_duration_ns",
    "thread_cpu_start_ns", "thread_cpu_end_ns", "thread_cpu_duration_ns",
    "process_cpu_start_ns", "process_cpu_end_ns", "process_cpu_duration_ns",
    "cpu_start", "cpu_end", "thread_nvcsw_delta", "thread_nivcsw_delta",
    "process_nvcsw_delta", "process_nivcsw_delta", "xfer_flags", "return_status",
    "diagnostic_flags", "overflow_count",
)


class ValidationError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise ValidationError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-trace", nargs="+", type=Path, required=True)
    parser.add_argument("--sdk-trace", nargs="+", type=Path, required=True)
    parser.add_argument("--baseline-trace", nargs="+", type=Path, required=True)
    parser.add_argument("--stock-app-trace", nargs="+", type=Path, required=True)
    parser.add_argument("--instrumented-app-trace", nargs="+", type=Path, required=True)
    parser.add_argument("--perf", nargs="+", type=Path, required=True)
    parser.add_argument("--expected-reps", type=int, default=30)
    parser.add_argument("--input-elements", type=int, default=2_621_440)
    parser.add_argument("--tasklets", type=int, default=16)
    parser.add_argument("--block-size-log2", type=int, default=10)
    parser.add_argument("--p50-threshold-pct", type=float, default=3.0)
    parser.add_argument("--p90-threshold-pct", type=float, default=5.0)
    parser.add_argument("--provenance-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csv(paths: Iterable[Path], header: tuple[str, ...], schema: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(paths):
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != header:
                fail(f"{path}: header differs from {schema}")
            file_rows = list(reader)
        if not file_rows:
            fail(f"{path}: trace is empty")
        for row in file_rows:
            if row["schema_version"] != schema:
                fail(f"{path}: schema_version={row['schema_version']}")
            row["_source_path"] = str(path)
        rows.extend(file_rows)
    return rows


def as_int(row: dict[str, str], field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, ValueError) as error:
        fail(f"{row.get('_source_path', 'row')}: invalid {field}: {error}")
    raise AssertionError


def aligned_per_dpu_elements(input_elements: int, actual_dpus: int) -> int:
    value = math.ceil(input_elements / actual_dpus)
    if value * 4 % 8:
        value = (value // 8) * 8 + 8
    return value


def validate_clock_interval(row: dict[str, str], prefix: str) -> None:
    start = as_int(row, f"{prefix}_start_ns")
    end = as_int(row, f"{prefix}_end_ns")
    duration = as_int(row, f"{prefix}_duration_ns")
    if end < start or duration != end - start:
        fail(f"{row['_source_path']}: invalid {prefix} interval")


def validate_app_rows(
    rows: list[dict[str, str]],
    input_elements: int,
    tasklets: int,
    block_size_log2: int,
    expected_reps: int | None = None,
) -> tuple[dict[tuple[str, int], dict[str, dict[str, str]]], dict[tuple[str, int], dict[str, int]]]:
    groups: dict[tuple[str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    run_ids: set[str] = set()
    for row in rows:
        for field in INTEGER_FIELDS_APP:
            as_int(row, field)
        if as_int(row, "input_elements") != input_elements:
            fail(f"{row['_source_path']}: input_elements differs")
        if as_int(row, "tasklets") != tasklets or as_int(row, "block_size_log2") != block_size_log2:
            fail(f"{row['_source_path']}: tasklet or block configuration differs")
        if as_int(row, "actual_dpus") <= 0 or as_int(row, "actual_ranks") != 1:
            fail(f"{row['_source_path']}: topology differs")
        if row["configuration"] not in ("single", "rank"):
            fail(f"{row['_source_path']}: configuration={row['configuration']}")
        if row["configuration"] == "single" and as_int(row, "actual_dpus") != 1:
            fail(f"{row['_source_path']}: single allocation has multiple DPUs")
        if row["configuration"] == "rank" and as_int(row, "actual_dpus") <= 1:
            fail(f"{row['_source_path']}: rank allocation lacks a full-rank topology")
        if as_int(row, "result_ok") != 1:
            fail(f"{row['_source_path']}: result_ok differs")
        component = row["component"]
        if component not in COMPONENT_META:
            fail(f"{row['_source_path']}: component={component}")
        if (row["direction"], row["target_space"], row["operation"]) != COMPONENT_META[component]:
            fail(f"{row['_source_path']}: metadata differs for {component}")
        for prefix in ("wall", "thread_cpu", "process_cpu"):
            validate_clock_interval(row, prefix)
        for field in (
            "thread_nvcsw_delta", "thread_nivcsw_delta", "process_nvcsw_delta",
            "process_nivcsw_delta",
        ):
            if as_int(row, field) < 0:
                fail(f"{row['_source_path']}: negative {field}")
        if as_int(row, "cpu_start") < 0 or as_int(row, "cpu_end") < 0:
            fail(f"{row['_source_path']}: invalid CPU ID")
        dpus = as_int(row, "actual_dpus")
        payload = aligned_per_dpu_elements(input_elements, dpus) * 4
        expected_bytes = 12 if component.endswith("ARGS") else payload
        if as_int(row, "bytes_per_dpu") != expected_bytes:
            fail(f"{row['_source_path']}: bytes_per_dpu differs for {component}")
        if as_int(row, "aggregate_bytes") != expected_bytes * dpus:
            fail(f"{row['_source_path']}: aggregate_bytes differs for {component}")
        for field in ("run_id", "source_sha256", "host_binary_sha256", "dpu_binary_sha256"):
            if row[field] in ("", "unknown"):
                fail(f"{row['_source_path']}: {field} lacks provenance")
        key = (row["run_id"], as_int(row, "repeat_id"))
        if component in groups[key]:
            fail(f"{row['_source_path']}: duplicate {key} {component}")
        groups[key][component] = row
        run_ids.add(row["run_id"])

    if expected_reps is not None and len(groups) != expected_reps:
        fail(f"expected {expected_reps} application samples, found {len(groups)}")
    if expected_reps is not None and len(run_ids) != expected_reps:
        fail(f"expected one run_id per process, found {len(run_ids)}")

    phase_totals: dict[tuple[str, int], dict[str, int]] = {}
    for key, indexed in groups.items():
        if set(indexed) != set(COMPONENTS):
            fail(f"{key}: component set differs")
        totals: dict[str, int] = {}
        for phase, components in (("H2D", H2D_COMPONENTS), ("D2H", D2H_COMPONENTS)):
            for left, right in zip(components, components[1:]):
                for prefix in ("wall", "thread_cpu", "process_cpu"):
                    if as_int(indexed[left], f"{prefix}_end_ns") != as_int(indexed[right], f"{prefix}_start_ns"):
                        fail(f"{key}: {prefix} boundary differs between {left} and {right}")
            totals[phase] = sum(as_int(indexed[name], "wall_duration_ns") for name in components)
        push_h2d = sum(as_int(indexed[name], "aggregate_bytes") for name in ("PUSH_ARGS", "PUSH_A", "PUSH_B"))
        push_d2h = as_int(indexed["PUSH_C"], "aggregate_bytes")
        totals["H2D_BYTES"] = push_h2d
        totals["D2H_BYTES"] = push_d2h
        phase_totals[key] = totals
    return dict(groups), phase_totals


def validate_baseline_crosscheck(
    paths: list[Path],
    app_groups: dict[tuple[str, int], dict[str, dict[str, str]]],
    phase_totals: dict[tuple[str, int], dict[str, int]],
) -> list[dict[str, str]]:
    rows = load_csv(paths, BASELINE_HEADER, "upmem.va_baseline.hw.v1")
    indexed: dict[tuple[str, int, str], dict[str, str]] = {}
    for row in rows:
        key = (row["run_id"], as_int(row, "repeat_id"), row["phase"])
        if key in indexed:
            fail(f"{row['_source_path']}: duplicate baseline row {key}")
        indexed[key] = row
    for run_key in app_groups:
        for phase in ("H2D", "D2H"):
            key = (*run_key, phase)
            if key not in indexed:
                fail(f"{run_key}: baseline lacks {phase}")
            row = indexed[key]
            if as_int(row, "duration_ns") != phase_totals[run_key][phase]:
                fail(f"{run_key}: {phase} component sum differs from baseline")
            if as_int(row, "aggregate_bytes") != phase_totals[run_key][f"{phase}_BYTES"]:
                fail(f"{run_key}: {phase} push bytes differ from baseline")
            if row["result_ok"] != "1" or row["expected_checksum"] != row["actual_checksum"]:
                fail(f"{run_key}: baseline functional result differs")
    return rows


def validate_sdk_rows(
    rows: list[dict[str, str]],
    app_groups: dict[tuple[str, int], dict[str, dict[str, str]]],
) -> list[dict[str, object]]:
    by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        for field in INTEGER_FIELDS_SDK:
            as_int(row, field)
        for prefix in ("wall", "thread_cpu", "process_cpu"):
            validate_clock_interval(row, prefix)
        if row["event_kind"] not in ("PUSH_CALL", "BACKEND_TRANSFER"):
            fail(f"{row['_source_path']}: event_kind={row['event_kind']}")
        if row["direction"] not in ("H2D", "D2H"):
            fail(f"{row['_source_path']}: direction={row['direction']}")
        if row["memory_type"] not in ("WRAM", "MRAM"):
            fail(f"{row['_source_path']}: memory_type={row['memory_type']}")
        if as_int(row, "return_status") != 0:
            fail(f"{row['_source_path']}: SDK return_status differs")
        if as_int(row, "diagnostic_flags") != 0:
            fail(f"{row['_source_path']}: SDK diagnostic_flags differs")
        if as_int(row, "overflow_count") != 0:
            fail(f"{row['_source_path']}: SDK trace overflow")
        if row["trace_clock"] != "CLOCK_MONOTONIC_RAW":
            fail(f"{row['_source_path']}: trace clock differs")
        if not row["runtime_version"].startswith("2025.1"):
            fail(f"{row['_source_path']}: runtime_version={row['runtime_version']}")
        if as_int(row, "pid") <= 0 or as_int(row, "tid") <= 0:
            fail(f"{row['_source_path']}: invalid process or thread ID")
        if as_int(row, "cpu_start") < 0 or as_int(row, "cpu_end") < 0:
            fail(f"{row['_source_path']}: invalid SDK CPU ID")
        if as_int(row, "rank_count") != 1 or as_int(row, "dpu_count") <= 0:
            fail(f"{row['_source_path']}: invalid SDK topology")
        for field in (
            "thread_nvcsw_delta", "thread_nivcsw_delta", "process_nvcsw_delta",
            "process_nivcsw_delta",
        ):
            if as_int(row, field) < 0:
                fail(f"{row['_source_path']}: negative SDK {field}")
        if row["event_kind"] == "PUSH_CALL" and as_int(row, "xfer_flags") != 0:
            fail(f"{row['_source_path']}: asynchronous or non-default transfer flags")
        if as_int(row, "aggregate_bytes") != as_int(row, "bytes_per_dpu") * as_int(row, "dpu_count"):
            fail(f"{row['_source_path']}: SDK byte conservation differs")
        by_run[row["run_id"]].append(row)

    app_by_run: dict[str, dict[str, dict[str, str]]] = {}
    for (run_id, _repeat), indexed in app_groups.items():
        if run_id in app_by_run:
            fail(f"{run_id}: multiple repetitions share one SDK run ID")
        app_by_run[run_id] = indexed
    if set(by_run) != set(app_by_run):
        fail(f"SDK run IDs differ from application run IDs")

    derived: list[dict[str, object]] = []
    expected_pushes = (
        ("PUSH_ARGS", "H2D", "WRAM"), ("PUSH_A", "H2D", "MRAM"),
        ("PUSH_B", "H2D", "MRAM"), ("PUSH_C", "D2H", "MRAM"),
    )
    for run_id, run_rows in by_run.items():
        pushes = sorted(
            (row for row in run_rows if row["event_kind"] == "PUSH_CALL"),
            key=lambda row: as_int(row, "wall_start_ns"),
        )
        if len(pushes) != len(expected_pushes):
            fail(f"{run_id}: expected four PUSH_CALL rows, found {len(pushes)}")
        backends = [row for row in run_rows if row["event_kind"] == "BACKEND_TRANSFER"]
        matched_backend_ids: set[int] = set()
        app_index = app_by_run[run_id]
        for push, (component, direction, memory_type) in zip(pushes, expected_pushes):
            app_row = app_index[component]
            if push["direction"] != direction or push["memory_type"] != memory_type:
                fail(f"{run_id}: SDK metadata differs for {component}")
            for field in ("bytes_per_dpu", "aggregate_bytes"):
                if as_int(push, field) != as_int(app_row, field):
                    fail(f"{run_id}: SDK {field} differs for {component}")
            if not (
                as_int(app_row, "wall_start_ns") <= as_int(push, "wall_start_ns")
                and as_int(push, "wall_end_ns") <= as_int(app_row, "wall_end_ns")
            ):
                fail(f"{run_id}: SDK PUSH_CALL escapes application {component}")
            if memory_type == "WRAM":
                continue
            candidates = [
                backend for backend in backends
                if id(backend) not in matched_backend_ids
                and backend["direction"] == direction
                and as_int(backend, "bytes_per_dpu") == as_int(push, "bytes_per_dpu")
                and as_int(backend, "aggregate_bytes") == as_int(push, "aggregate_bytes")
                and as_int(push, "wall_start_ns") <= as_int(backend, "wall_start_ns")
                and as_int(backend, "wall_end_ns") <= as_int(push, "wall_end_ns")
            ]
            if len(candidates) != 1:
                fail(f"{run_id}: {component} has {len(candidates)} matching backend intervals")
            backend = candidates[0]
            matched_backend_ids.add(id(backend))
            pre = as_int(backend, "wall_start_ns") - as_int(push, "wall_start_ns")
            backend_ns = as_int(backend, "wall_duration_ns")
            tail = as_int(push, "wall_end_ns") - as_int(backend, "wall_end_ns")
            if pre + backend_ns + tail != as_int(push, "wall_duration_ns"):
                fail(f"{run_id}: SDK interval decomposition differs for {component}")
            derived.append({
                "run_id": run_id,
                "component": component,
                "direction": direction,
                "bytes_per_dpu": as_int(push, "bytes_per_dpu"),
                "aggregate_bytes": as_int(push, "aggregate_bytes"),
                "c_pre_wall_ns": pre,
                "t_backend_wall_ns": backend_ns,
                "c_tail_wall_ns": tail,
                "backend_process_cpu_ns": as_int(backend, "process_cpu_duration_ns"),
            })
    return derived


def percentile(values: list[int], percent: float) -> float:
    if not values:
        fail("percentile input is empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def compare_overhead(
    stock_totals: dict[tuple[str, int], dict[str, int]],
    instrumented_totals: dict[tuple[str, int], dict[str, int]],
    p50_threshold: float,
    p90_threshold: float,
) -> tuple[list[dict[str, object]], bool]:
    if len(stock_totals) < 2 or len(instrumented_totals) < 2:
        fail("overhead comparison requires at least two processes per library")
    output: list[dict[str, object]] = []
    passed = True
    for phase in ("H2D", "D2H"):
        stock = [sample[phase] for sample in stock_totals.values()]
        instrumented = [sample[phase] for sample in instrumented_totals.values()]
        for label, value, threshold in (("p50", 50.0, p50_threshold), ("p90", 90.0, p90_threshold)):
            stock_ns = percentile(stock, value)
            instrumented_ns = percentile(instrumented, value)
            shift = abs(instrumented_ns - stock_ns) * 100.0 / stock_ns
            status = "PASS" if shift <= threshold else "FAIL"
            passed = passed and status == "PASS"
            output.append({
                "phase": phase,
                "percentile": label,
                "stock_ns": f"{stock_ns:.3f}",
                "instrumented_ns": f"{instrumented_ns:.3f}",
                "absolute_shift_pct": f"{shift:.6f}",
                "threshold_pct": f"{threshold:.6f}",
                "status": status,
            })
    return output, passed


def validate_perf(paths: list[Path], expected_reps: int) -> dict[str, object]:
    if len(paths) != expected_reps:
        fail(f"expected {expected_reps} perf files, found {len(paths)}")
    required = {"task-clock", "cycles", "instructions"}
    evidence: dict[str, dict[str, int]] = {}
    for path in sorted(paths):
        found: dict[str, int] = {}
        for line in path.read_text().splitlines():
            fields = line.split(",")
            if len(fields) < 3:
                continue
            event = fields[2].strip()
            if event not in required:
                continue
            value_text = fields[0].strip().replace(" ", "")
            try:
                value = int(float(value_text))
            except ValueError:
                fail(f"{path}: invalid perf value for {event}: {fields[0]}")
            if value <= 0:
                fail(f"{path}: perf value for {event} must be positive")
            found[event] = value
        if set(found) != required:
            fail(f"{path}: required perf events are {sorted(found)}")
        evidence[path.name] = found
    return {"status": "PASS", "files": evidence}


def validate_provenance(root: Path) -> dict[str, object]:
    required = (
        "uname.txt", "lscpu.txt", "numa.txt", "numactl_show.txt", "config.txt",
        "prim_git_commit.txt", "prim_git_status.txt", "source.sha256", "binaries.sha256",
        "sdk_source_baseline_check.txt", "sdk_source.sha256", "sdk_patch_check.txt",
        "sdk_build.log", "instrumented_library.sha256", "dynamic_library_resolution.txt",
        "imc_status.txt",
    )
    files: dict[str, dict[str, object]] = {}
    for name in required:
        path = root / name
        if not path.is_file():
            fail(f"provenance file is absent: {path}")
        if path.stat().st_size == 0 and name != "prim_git_status.txt":
            fail(f"provenance file is empty: {path}")
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    dynamic = (root / "dynamic_library_resolution.txt").read_text()
    if str(root.resolve()) not in dynamic or "libdpu.so" not in dynamic:
        fail("dynamic library evidence does not resolve libdpu inside RESULT_ROOT")
    baseline_check = (root / "sdk_source_baseline_check.txt").read_text()
    if "OK" not in baseline_check:
        fail("SDK source baseline hash check lacks OK records")
    patch_check = (root / "sdk_patch_check.txt").read_text()
    if "PATCH_APPLIED=PASS" not in patch_check:
        fail("SDK patch evidence differs")
    return {"status": "PASS", "root": str(root.resolve()), "files": files}


def write_csv(path: Path, fieldnames: tuple[str, ...] | list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def clean_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]


def summary_rows(app_rows: list[dict[str, str]], derived: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    by_component: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in app_rows:
        by_component[row["component"]].append(row)
    for component in COMPONENTS:
        rows = by_component[component]
        for metric in ("wall_duration_ns", "thread_cpu_duration_ns", "process_cpu_duration_ns"):
            values = [as_int(row, metric) for row in rows]
            output.append({
                "layer": "APP", "item": component, "direction": COMPONENT_META[component][0],
                "metric": metric, "samples": len(values), "minimum": min(values),
                "p50": f"{percentile(values, 50):.3f}", "p90": f"{percentile(values, 90):.3f}",
                "maximum": max(values),
            })
    for component in ("PUSH_A", "PUSH_B", "PUSH_C"):
        rows = [row for row in derived if row["component"] == component]
        for metric in ("c_pre_wall_ns", "t_backend_wall_ns", "c_tail_wall_ns", "backend_process_cpu_ns"):
            values = [int(row[metric]) for row in rows]
            output.append({
                "layer": "SDK", "item": component, "direction": rows[0]["direction"],
                "metric": metric, "samples": len(values), "minimum": min(values),
                "p50": f"{percentile(values, 50):.3f}", "p90": f"{percentile(values, 90):.3f}",
                "maximum": max(values),
            })
    return output


def main() -> int:
    args = parse_args()
    try:
        app_rows = load_csv(args.app_trace, APP_HEADER, APP_SCHEMA)
        sdk_rows = load_csv(args.sdk_trace, SDK_HEADER, SDK_SCHEMA)
        stock_rows = load_csv(args.stock_app_trace, APP_HEADER, APP_SCHEMA)
        instrumented_rows = load_csv(args.instrumented_app_trace, APP_HEADER, APP_SCHEMA)
        app_groups, phase_totals = validate_app_rows(
            app_rows, args.input_elements, args.tasklets, args.block_size_log2,
            args.expected_reps,
        )
        _stock_groups, stock_totals = validate_app_rows(
            stock_rows, args.input_elements, args.tasklets, args.block_size_log2,
        )
        _instrumented_groups, instrumented_totals = validate_app_rows(
            instrumented_rows, args.input_elements, args.tasklets, args.block_size_log2,
        )
        if any(row["configuration"] != "rank" for row in app_rows + stock_rows + instrumented_rows):
            fail("formal and overhead collections require rank allocation")
        formal_dpus = {as_int(row, "actual_dpus") for row in app_rows}
        stock_dpus = {as_int(row, "actual_dpus") for row in stock_rows}
        instrumented_dpus = {as_int(row, "actual_dpus") for row in instrumented_rows}
        if len(formal_dpus) != 1 or formal_dpus != stock_dpus or formal_dpus != instrumented_dpus:
            fail("rank topology changed across formal and overhead collections")
        baseline_rows = validate_baseline_crosscheck(args.baseline_trace, app_groups, phase_totals)
        derived = validate_sdk_rows(sdk_rows, app_groups)
        overhead_rows, overhead_passed = compare_overhead(
            stock_totals, instrumented_totals, args.p50_threshold_pct, args.p90_threshold_pct,
        )
        perf = validate_perf(args.perf, args.expected_reps)
        provenance = validate_provenance(args.provenance_root)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_dir / "va_transfer_breakdown_events.csv", APP_HEADER, clean_rows(app_rows))
        write_csv(args.output_dir / "sdk_transfer_events.csv", SDK_HEADER, clean_rows(sdk_rows))
        write_csv(args.output_dir / "va_baseline_events.csv", BASELINE_HEADER, clean_rows(baseline_rows))
        write_csv(
            args.output_dir / "sdk_interval_breakdown.csv",
            ["run_id", "component", "direction", "bytes_per_dpu", "aggregate_bytes",
             "c_pre_wall_ns", "t_backend_wall_ns", "c_tail_wall_ns", "backend_process_cpu_ns"],
            derived,
        )
        write_csv(
            args.output_dir / "transfer_breakdown_summary.csv",
            ["layer", "item", "direction", "metric", "samples", "minimum", "p50", "p90", "maximum"],
            summary_rows(app_rows, derived),
        )
        write_csv(
            args.output_dir / "overhead_comparison.csv",
            ["phase", "percentile", "stock_ns", "instrumented_ns", "absolute_shift_pct",
             "threshold_pct", "status"],
            overhead_rows,
        )
        manifest = {
            "schema_version": "upmem.va_transfer_breakdown_manifest.v1",
            "functional_status": "PASS",
            "timing_status": "REPORTED" if overhead_passed else "FAIL",
            "formal_processes": len(app_groups),
            "app_rows": len(app_rows),
            "sdk_rows": len(sdk_rows),
            "sdk_matched_mram_pushes": len(derived),
            "overhead": overhead_rows,
            "perf": perf,
            "provenance": provenance,
        }
        (args.output_dir / "va_transfer_breakdown_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        if not overhead_passed:
            print("FAIL VA_TRANSFER_BREAKDOWN overhead threshold exceeded")
            return 1
    except (OSError, ValueError, ValidationError) as error:
        print(f"FAIL VA_TRANSFER_BREAKDOWN {error}")
        return 1
    print(
        f"PASS VA_TRANSFER_BREAKDOWN functional_status=PASS timing_status=REPORTED "
        f"processes={len(app_groups)} output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
