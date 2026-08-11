#!/usr/bin/env python3
"""Evaluate SCAN context-key candidates and diagnose known outliers."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Tuple

from analyze_transport_keys import group_statistics, percentile
from transport_key import (
    TRANSFER_OPS,
    previous_sdk_op_class,
    transport_key,
    transport_key_v8,
)


Row = Dict[str, str]
Sample = Tuple[Path, Row]
KeyFunction = Callable[[Row], str]


def base_key(row: Row) -> str:
    return transport_key_v8(row)


def warmup_key(row: Row) -> str:
    return f"{transport_key_v8(row)};warmup={row['warmup']}"


def predecessor_key(row: Row) -> str:
    return (
        f"{transport_key_v8(row)};"
        f"previous_sdk_op_class={previous_sdk_op_class(row)}"
    )


def reuse_key(row: Row) -> str:
    return (
        f"{transport_key_v8(row)};"
        f"source_buffer_reuse_class={row['source_buffer_reuse_class']};"
        f"target_region_reuse_class={row['target_region_reuse_class']}"
    )


MODEL_KEYS: dict[str, KeyFunction] = {
    "base_v8": base_key,
    "warmup_v8": warmup_key,
    "previous_sdk_op_class_v8": predecessor_key,
    "reuse_context_v8": reuse_key,
    "base_v9": lambda row: row["transport_key"],
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_samples(paths: list[Path]) -> list[Sample]:
    samples: list[Sample] = []
    for path in paths:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"{path}: empty trace")
        for row in rows:
            if row["op"] not in TRANSFER_OPS:
                continue
            if row["transport_key"] != transport_key(row):
                raise ValueError(
                    f"{path}: event {row.get('event_id', '?')} has invalid "
                    "transport_key"
                )
            samples.append((path, row))
    if not samples:
        raise ValueError("input traces contain zero transfer events")
    return samples


def context_value(model: str, row: Row) -> str:
    if model == "warmup_v8":
        return "WARMUP" if row["warmup"] == "1" else "ITERATIVE"
    if model == "previous_sdk_op_class_v8":
        return previous_sdk_op_class(row)
    if model == "reuse_context_v8":
        return (
            f"{row['source_buffer_reuse_class']}|"
            f"{row['target_region_reuse_class']}"
        )
    if model == "base_v9":
        return (
            f"{previous_sdk_op_class(row)}|"
            f"{row['source_buffer_reuse_class']}|"
            f"{row['target_region_reuse_class']}"
        )
    return "POOLED"


def summarize_context_groups(
    samples: list[Sample],
    min_samples: int,
    min_traces: int,
    spread_threshold_pct: float,
    cv_threshold_pct: float,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    base_group_count = len({transport_key_v8(row) for _, row in samples})
    for model, key_function in MODEL_KEYS.items():
        groups: dict[str, list[Sample]] = defaultdict(list)
        for sample in samples:
            groups[key_function(sample[1])].append(sample)
        group_count = len(groups)
        growth_pct = 100.0 * (group_count - base_group_count) / base_group_count
        for key, group in sorted(groups.items()):
            row = group[0][1]
            result.append(
                {
                    "model": model,
                    "candidate_key": key,
                    "context_value": context_value(model, row),
                    "subop": row["subop"],
                    "active_dpus": row["active_dpus"],
                    "direction": row["direction"],
                    "target_space": row["target_space"],
                    "transfer_bytes_per_dpu": row["transfer_bytes_per_dpu"],
                    "table_key_count": group_count,
                    "base_key_count": base_group_count,
                    "table_growth_pct": f"{growth_pct:.3f}",
                    **group_statistics(
                        group,
                        min_samples,
                        min_traces,
                        spread_threshold_pct,
                        cv_threshold_pct,
                    ),
                }
            )
    return result


def event_scopes(row: Row) -> list[str]:
    scopes = ["ALL", row["subop"]]
    if row["subop"] in {"input_arguments_scan", "input_arguments_add"}:
        scopes.append(
            f"{row['subop']}:{previous_sdk_op_class(row)}"
        )
    return scopes


def error_summary(
    actual: list[int], predicted: list[float]
) -> dict[str, str]:
    absolute_pct = [
        100.0 * abs(estimate - value) / value
        for value, estimate in zip(actual, predicted)
    ]
    signed_pct = [
        100.0 * (estimate - value) / value
        for value, estimate in zip(actual, predicted)
    ]
    return {
        "median_abs_pct_error": f"{statistics.median(absolute_pct):.6f}",
        "mean_abs_pct_error": f"{statistics.mean(absolute_pct):.6f}",
        "p90_abs_pct_error": f"{percentile(absolute_pct, 0.90):.6f}",
        "p95_abs_pct_error": f"{percentile(absolute_pct, 0.95):.6f}",
        "mean_signed_pct_error": f"{statistics.mean(signed_pct):.6f}",
    }


def evaluate_leave_one_trace_out(
    paths: list[Path], samples: list[Sample]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    samples_by_path: dict[Path, list[Row]] = defaultdict(list)
    for path, row in samples:
        samples_by_path[path].append(row)
    if len(samples_by_path) < 2:
        raise ValueError("leave-one-trace-out evaluation requires at least 2 traces")

    base_group_count = len({transport_key_v8(row) for _, row in samples})
    pooled: dict[tuple[str, str], tuple[list[int], list[float], int]] = {}
    per_trace: list[dict[str, object]] = []

    for model, key_function in MODEL_KEYS.items():
        for held_out_path in sorted(samples_by_path):
            training: dict[str, list[int]] = defaultdict(list)
            for path, rows in samples_by_path.items():
                if path == held_out_path:
                    continue
                for row in rows:
                    training[key_function(row)].append(int(row["measured_ns"]))
            lookup = {
                key: float(statistics.median(values))
                for key, values in training.items()
            }
            held_out_by_scope: dict[str, list[Row]] = defaultdict(list)
            for row in samples_by_path[held_out_path]:
                for scope in event_scopes(row):
                    held_out_by_scope[scope].append(row)
            for scope, rows in sorted(held_out_by_scope.items()):
                actual: list[int] = []
                predicted: list[float] = []
                for row in rows:
                    estimate = lookup.get(key_function(row))
                    if estimate is None:
                        continue
                    actual.append(int(row["measured_ns"]))
                    predicted.append(estimate)
                summary: dict[str, object] = {
                    "model": model,
                    "scope": scope,
                    "held_out_trace": str(held_out_path),
                    "training_key_count": len(lookup),
                    "evaluation_rows": len(rows),
                    "predicted_rows": len(predicted),
                    "coverage_pct": (
                        "0.000000"
                        if not rows
                        else f"{100.0 * len(predicted) / len(rows):.6f}"
                    ),
                    "median_abs_pct_error": "",
                    "mean_abs_pct_error": "",
                    "p90_abs_pct_error": "",
                    "p95_abs_pct_error": "",
                    "mean_signed_pct_error": "",
                }
                if predicted:
                    summary.update(error_summary(actual, predicted))
                per_trace.append(summary)
                key = (model, scope)
                pooled_actual, pooled_predicted, total = pooled.get(
                    key, ([], [], 0)
                )
                pooled_actual.extend(actual)
                pooled_predicted.extend(predicted)
                pooled[key] = (
                    pooled_actual,
                    pooled_predicted,
                    total + len(rows),
                )

    result: list[dict[str, object]] = []
    for (model, scope), (actual, predicted, total) in sorted(pooled.items()):
        key_function = MODEL_KEYS[model]
        full_key_count = len({key_function(row) for _, row in samples})
        growth_pct = 100.0 * (
            full_key_count - base_group_count
        ) / base_group_count
        summary = {
            "model": model,
            "scope": scope,
            "trace_files": len(paths),
            "table_key_count": full_key_count,
            "base_key_count": base_group_count,
            "table_growth_pct": f"{growth_pct:.3f}",
            "evaluation_rows": total,
            "predicted_rows": len(predicted),
            "coverage_pct": (
                "0.000000"
                if not total
                else f"{100.0 * len(predicted) / total:.6f}"
            ),
            "median_abs_pct_error": "",
            "mean_abs_pct_error": "",
            "p90_abs_pct_error": "",
            "p95_abs_pct_error": "",
            "mean_signed_pct_error": "",
        }
        if predicted:
            summary.update(error_summary(actual, predicted))
        result.append(summary)
    return result, per_trace


def load_schedule(path: Path | None) -> dict[tuple[str, str], Row]:
    if path is None:
        return {}
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {
        (row["config"], row["round"]): row
        for row in rows
        if row["phase"] == "trace"
    }


def anomaly_diagnostics(
    samples: list[Sample], schedule_path: Path | None
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    targets = {
        (subop, scale)
        for subop in (
            "input_arguments_scan", "input_data", "partial_results",
            "input_arguments_add", "output_data",
        )
        for scale in ("64", "128", "256", "512", "1024", "1216")
    }
    groups: dict[tuple[str, str], list[Sample]] = defaultdict(list)
    for sample in samples:
        row = sample[1]
        target = (row["subop"], row["active_dpus"])
        if target in targets:
            groups[target].append(sample)
    schedule = load_schedule(schedule_path)
    events: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    for (subop, active_dpus), group in sorted(groups.items()):
        durations = [int(row["measured_ns"]) for _, row in group]
        median = float(statistics.median(durations))
        q1 = percentile(durations, 0.25)
        q3 = percentile(durations, 0.75)
        iqr = q3 - q1
        mad = float(statistics.median(abs(value - median) for value in durations))
        high_fence = q3 + 3.0 * iqr
        low_fence = q1 - 3.0 * iqr
        ranked = sorted(
            group,
            key=lambda sample: int(sample[1]["measured_ns"]),
            reverse=True,
        )
        for rank, (path, row) in enumerate(ranked, start=1):
            measured = int(row["measured_ns"])
            robust_z = (
                0.0 if mad == 0 else 0.6745 * (measured - median) / mad
            )
            schedule_row = schedule.get((row["run_id"], row["repeat_id"]), {})
            events.append(
                {
                    "subop": subop,
                    "active_dpus": active_dpus,
                    "descending_rank": rank,
                    "trace_path": str(path),
                    "run_id": row["run_id"],
                    "repeat_id": row["repeat_id"],
                    "iteration": row["iteration"],
                    "warmup": row["warmup"],
                    "previous_sdk_op": row["previous_sdk_op"],
                    "measured_ns": measured,
                    "group_median_ns": round(median),
                    "ratio_to_median": f"{measured / median:.6f}",
                    "robust_z": f"{robust_z:.6f}",
                    "extreme_tukey_outlier": int(
                        measured > high_fence or measured < low_fence
                    ),
                    "schedule_round": schedule_row.get("round", ""),
                    "schedule_slot": schedule_row.get("slot", ""),
                    "schedule_start_wall_ns": schedule_row.get(
                        "start_wall_ns", ""
                    ),
                    "schedule_end_wall_ns": schedule_row.get("end_wall_ns", ""),
                }
            )

        by_path: dict[Path, list[Row]] = defaultdict(list)
        for path, row in group:
            by_path[path].append(row)
        for path, rows in sorted(by_path.items()):
            values = [int(row["measured_ns"]) for row in rows]
            first = rows[0]
            schedule_row = schedule.get(
                (first["run_id"], first["repeat_id"]), {}
            )
            traces.append(
                {
                    "subop": subop,
                    "active_dpus": active_dpus,
                    "trace_path": str(path),
                    "run_id": first["run_id"],
                    "repeat_id": first["repeat_id"],
                    "sample_count": len(values),
                    "median_ns": round(statistics.median(values)),
                    "min_ns": min(values),
                    "max_ns": max(values),
                    "max_ratio_to_global_median": f"{max(values) / median:.6f}",
                    "extreme_event_count": sum(
                        value > high_fence or value < low_fence
                        for value in values
                    ),
                    "schedule_round": schedule_row.get("round", ""),
                    "schedule_slot": schedule_row.get("slot", ""),
                    "schedule_start_wall_ns": schedule_row.get(
                        "start_wall_ns", ""
                    ),
                    "schedule_end_wall_ns": schedule_row.get("end_wall_ns", ""),
                }
            )
    return events, traces


def analyze(
    paths: list[Path],
    min_samples: int,
    min_traces: int,
    spread_threshold_pct: float,
    cv_threshold_pct: float,
    schedule_path: Path | None = None,
) -> dict[str, list[dict[str, object]]]:
    samples = load_samples(paths)
    groups = summarize_context_groups(
        samples,
        min_samples,
        min_traces,
        spread_threshold_pct,
        cv_threshold_pct,
    )
    holdout, per_trace = evaluate_leave_one_trace_out(paths, samples)
    anomaly_events, anomaly_traces = anomaly_diagnostics(samples, schedule_path)
    return {
        "context_group_summary": groups,
        "holdout_summary": holdout,
        "holdout_per_trace": per_trace,
        "anomaly_events": anomaly_events,
        "anomaly_trace_summary": anomaly_traces,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--min-traces", type=int, default=20)
    parser.add_argument("--spread-threshold-pct", type=float, default=25.0)
    parser.add_argument("--cv-threshold-pct", type=float, default=25.0)
    args = parser.parse_args()
    if args.min_samples < 2 or args.min_traces < 2:
        parser.error("sample and trace thresholds must be at least 2")
    if args.spread_threshold_pct < 0 or args.cv_threshold_pct < 0:
        parser.error("stability thresholds must be non-negative")
    if args.schedule is not None and not args.schedule.is_file():
        parser.error(f"schedule does not exist: {args.schedule}")

    try:
        outputs = analyze(
            args.traces,
            args.min_samples,
            args.min_traces,
            args.spread_threshold_pct,
            args.cv_threshold_pct,
            args.schedule,
        )
    except ValueError as error:
        parser.error(str(error))

    for name, rows in outputs.items():
        path = args.output_dir / f"{name}.csv"
        if rows:
            write_csv(path, rows)
            print(f"{name}={path}")
        else:
            print(f"{name}=skipped_no_matching_rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
