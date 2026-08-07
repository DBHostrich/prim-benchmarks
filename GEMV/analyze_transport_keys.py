#!/usr/bin/env python3
"""Summarize repeated GEMV transfer costs by v8 key."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from transport_key import TRANSFER_OPS, transport_key


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def group_statistics(
    samples: list[tuple[Path, dict[str, str]]],
    min_samples: int,
    min_traces: int,
    spread_threshold_pct: float,
    cv_threshold_pct: float,
) -> dict[str, object]:
    durations = [int(row["measured_ns"]) for _, row in samples]
    median = float(statistics.median(durations))
    p10 = percentile(durations, 0.10)
    p90 = percentile(durations, 0.90)
    mean = float(statistics.mean(durations))
    stdev = float(statistics.stdev(durations)) if len(durations) > 1 else 0.0
    spread_pct = 0.0 if median == 0 else 100.0 * (p90 - p10) / median
    cv_pct = 0.0 if mean == 0 else 100.0 * stdev / mean
    trace_count = len({str(path) for path, _ in samples})
    failures: list[str] = []
    if len(durations) < min_samples:
        failures.append(f"samples<{min_samples}")
    if trace_count < min_traces:
        failures.append(f"traces<{min_traces}")
    if failures:
        status = "insufficient"
    else:
        if spread_pct > spread_threshold_pct:
            failures.append(f"spread>{spread_threshold_pct:g}%")
        if cv_pct > cv_threshold_pct:
            failures.append(f"cv>{cv_threshold_pct:g}%")
        status = "stable" if not failures else "unstable"
    samples_per_trace = Counter(str(path) for path, _ in samples)
    return {
        "trace_count": trace_count,
        "sample_count": len(durations),
        "samples_per_trace_min": min(samples_per_trace.values()),
        "samples_per_trace_max": max(samples_per_trace.values()),
        "median_ns": round(median),
        "p10_ns": round(p10),
        "p90_ns": round(p90),
        "p90_p10_spread_pct": f"{spread_pct:.3f}",
        "mean_ns": f"{mean:.3f}",
        "stdev_ns": f"{stdev:.3f}",
        "cv_pct": f"{cv_pct:.3f}",
        "min_ns": min(durations),
        "max_ns": max(durations),
        "status": status,
        "status_reason": "within_thresholds" if not failures else "|".join(failures),
    }


def joined_values(
    samples: list[tuple[Path, dict[str, str]]],
    field: str,
    numeric: bool = False,
) -> str:
    values = {row[field] for _, row in samples if row.get(field, "")}
    return "|".join(sorted(values, key=int if numeric else None))


def analyze(
    paths: list[Path],
    min_samples: int,
    min_traces: int,
    spread_threshold_pct: float,
    cv_threshold_pct: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    groups: dict[str, list[tuple[Path, dict[str, str]]]] = defaultdict(list)
    transfer_rows = 0
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
                    f"{path}: event {row.get('event_id', '?')} has invalid transport_key"
                )
            groups[row["transport_key"]].append((path, row))
            transfer_rows += 1

    summaries: list[dict[str, object]] = []
    for key, samples in sorted(groups.items()):
        row = samples[0][1]
        warmup_durations = [
            int(sample["measured_ns"])
            for _, sample in samples
            if sample["warmup"] == "1"
        ]
        iterative_durations = [
            int(sample["measured_ns"])
            for _, sample in samples
            if sample["warmup"] == "0"
        ]
        warmup_median = (
            float(statistics.median(warmup_durations))
            if warmup_durations
            else 0.0
        )
        iterative_median = (
            float(statistics.median(iterative_durations))
            if iterative_durations
            else 0.0
        )
        summaries.append(
            {
                "transport_key": key,
                "op": row["op"],
                "direction": row["direction"],
                "sdk_api_kind": row["sdk_api_kind"],
                "timing_scope": row["timing_scope"],
                "logical_distribution_class": row["logical_distribution_class"],
                "target_space": row["target_space"],
                "transfer_bytes_per_dpu": row["transfer_bytes_per_dpu"],
                "active_dpus": row["active_dpus"],
                "active_ranks": row["active_ranks"],
                "active_dpus_per_rank": row["active_dpus_per_rank"],
                "same_source_across_group": row["same_source_across_group"],
                "host_numa_node": row["host_numa_node"],
                "dpu_rank_numa_nodes": row["dpu_rank_numa_nodes"],
                "cpu_dpu_numa_relation": row["cpu_dpu_numa_relation"],
                "dpu_channel_ids": row["dpu_channel_ids"],
                "dpu_sysfs_rank_ids": row["dpu_sysfs_rank_ids"],
                "allocated_topology_signature": row[
                    "allocated_topology_signature"
                ],
                "allocated_dpus": row["allocated_dpus"],
                "allocated_ranks": row["allocated_ranks"],
                "previous_sdk_mux_domain_class": row[
                    "previous_sdk_mux_domain_class"
                ],
                "configured_dpus_values": joined_values(
                    samples, "configured_dpus", numeric=True
                ),
                "num_tasklets_values": joined_values(
                    samples, "num_tasklets", numeric=True
                ),
                "subops": joined_values(samples, "subop"),
                "phase_classes": joined_values(samples, "phase_class"),
                "warmup_values": joined_values(samples, "warmup", numeric=True),
                "warmup_sample_count": len(warmup_durations),
                "warmup_median_ns": round(warmup_median),
                "iterative_sample_count": len(iterative_durations),
                "iterative_median_ns": round(iterative_median),
                "warmup_vs_iterative_median_delta_pct": (
                    ""
                    if not warmup_durations
                    or not iterative_durations
                    or iterative_median == 0
                    else f"{100.0 * (warmup_median - iterative_median) / iterative_median:.3f}"
                ),
                "iterations": joined_values(samples, "iteration", numeric=True),
                "total_logical_bytes_values": joined_values(
                    samples, "total_logical_bytes", numeric=True
                ),
                "total_transfer_bytes_values": joined_values(
                    samples, "total_transfer_bytes", numeric=True
                ),
                "target_symbols": joined_values(samples, "target_symbol"),
                "offset_bytes_values": joined_values(
                    samples, "offset_bytes", numeric=True
                ),
                "run_repeat_count": len(
                    {
                        (sample.get("run_id", ""), sample["repeat_id"])
                        for _, sample in samples
                    }
                ),
                **group_statistics(
                    samples,
                    min_samples,
                    min_traces,
                    spread_threshold_pct,
                    cv_threshold_pct,
                ),
            }
        )

    statuses = Counter(str(row["status"]) for row in summaries)
    samples_by_status = Counter()
    for row in summaries:
        samples_by_status[str(row["status"])] += int(row["sample_count"])
    eligible_groups = statuses["stable"] + statuses["unstable"]
    eligible_samples = samples_by_status["stable"] + samples_by_status["unstable"]
    overview: dict[str, object] = {
        "trace_files": len(paths),
        "transfer_rows": transfer_rows,
        "transport_key_groups": len(summaries),
        "stable_groups": statuses["stable"],
        "unstable_groups": statuses["unstable"],
        "insufficient_groups": statuses["insufficient"],
        "stable_samples": samples_by_status["stable"],
        "unstable_samples": samples_by_status["unstable"],
        "insufficient_samples": samples_by_status["insufficient"],
        "stable_transport_key_pct": (
            "0.000"
            if eligible_groups == 0
            else f"{100.0 * statuses['stable'] / eligible_groups:.3f}"
        ),
        "stable_event_pct": (
            "0.000"
            if eligible_samples == 0
            else f"{100.0 * samples_by_status['stable'] / eligible_samples:.3f}"
        ),
        "min_samples": min_samples,
        "min_traces": min_traces,
        "spread_threshold_pct": spread_threshold_pct,
        "cv_threshold_pct": cv_threshold_pct,
    }
    return summaries, overview


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--min-traces", type=int, default=20)
    parser.add_argument("--spread-threshold-pct", type=float, default=25.0)
    parser.add_argument("--cv-threshold-pct", type=float, default=25.0)
    args = parser.parse_args()
    if args.min_samples < 2 or args.min_traces < 2:
        parser.error("sample and trace thresholds must be at least 2")
    if args.spread_threshold_pct < 0 or args.cv_threshold_pct < 0:
        parser.error("stability thresholds must be non-negative")

    summaries, overview = analyze(
        args.traces,
        args.min_samples,
        args.min_traces,
        args.spread_threshold_pct,
        args.cv_threshold_pct,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if summaries:
        with args.output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    for key, value in overview.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
