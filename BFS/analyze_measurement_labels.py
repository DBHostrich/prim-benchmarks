#!/usr/bin/env python3
"""Test whether repeated BFS transfers with the same ten-field label are stable."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


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


def analyze(
    paths: list[Path], min_samples: int, spread_threshold_pct: float
) -> tuple[list[dict[str, object]], dict[str, object]]:
    groups: dict[tuple[int, int, str], list[tuple[Path, dict[str, str]]]] = defaultdict(list)
    transfer_rows = 0

    for path in paths:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"{path}: empty trace")
        seen_in_trace: set[tuple[int, int, str]] = set()
        for row in rows:
            if row["op"] not in {"dpu_copy_to", "dpu_copy_from"}:
                continue
            transfer_rows += 1
            key = (
                int(row["configured_dpus"]),
                int(row["num_tasklets"]),
                row["measurement_label"],
            )
            if key in seen_in_trace:
                raise ValueError(
                    f"{path}: duplicate measurement_label within one controlled run: "
                    f"{row['measurement_label']}"
                )
            seen_in_trace.add(key)
            groups[key].append((path, row))

    summaries: list[dict[str, object]] = []
    for (configured_dpus, num_tasklets, label), samples in sorted(groups.items()):
        representative = samples[0][1]
        durations = [int(row["measured_ns"]) for _, row in samples]
        median = float(statistics.median(durations))
        p10 = percentile(durations, 0.10)
        p90 = percentile(durations, 0.90)
        mean = float(statistics.mean(durations))
        stdev = float(statistics.stdev(durations)) if len(durations) > 1 else 0.0
        spread_pct = 0.0 if median == 0 else 100.0 * (p90 - p10) / median
        cv_pct = 0.0 if mean == 0 else 100.0 * stdev / mean
        if len(durations) < min_samples:
            status = "insufficient"
        elif spread_pct <= spread_threshold_pct:
            status = "stable"
        else:
            status = "unstable"
        summaries.append(
            {
                "configured_dpus": configured_dpus,
                "num_tasklets": num_tasklets,
                "measurement_label": label,
                "op": representative["op"],
                "direction": representative["direction"],
                "api_type": representative["api_type"],
                "logical_distribution_class": representative[
                    "logical_distribution_class"
                ],
                "same_source_across_group": representative[
                    "same_source_across_group"
                ],
                "target_space": representative["target_space"],
                "transfer_bytes": representative["transfer_bytes"],
                "rank_ordinal": representative["rank_ordinal"],
                "dpu_id_in_rank": representative["dpu_id_in_rank"],
                "offset_feature": representative["offset_feature"],
                "call_context": representative["call_context"],
                "host_numa_node": representative["host_numa_node"],
                "subops": "|".join(sorted({row["subop"] for _, row in samples})),
                "sample_count": len(durations),
                "repeat_count": len({row["repeat_id"] for _, row in samples}),
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
            }
        )

    status_counts = Counter(row["status"] for row in summaries)
    sample_status_counts = Counter()
    for row in summaries:
        sample_status_counts[str(row["status"])] += int(row["sample_count"])
    overview: dict[str, object] = {
        "trace_files": len(paths),
        "transfer_rows": transfer_rows,
        "controlled_label_groups": len(summaries),
        "stable_groups": status_counts["stable"],
        "unstable_groups": status_counts["unstable"],
        "insufficient_groups": status_counts["insufficient"],
        "stable_samples": sample_status_counts["stable"],
        "unstable_samples": sample_status_counts["unstable"],
        "insufficient_samples": sample_status_counts["insufficient"],
        "min_samples": min_samples,
        "spread_threshold_pct": spread_threshold_pct,
    }
    return summaries, overview


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--spread-threshold-pct", type=float, default=25.0)
    args = parser.parse_args()
    if args.min_samples < 2:
        parser.error("--min-samples must be at least 2")
    if args.spread_threshold_pct < 0:
        parser.error("--spread-threshold-pct must be non-negative")

    summaries, overview = analyze(
        args.traces, args.min_samples, args.spread_threshold_pct
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not summaries:
        raise SystemExit("No dpu_copy_to/from rows found")
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    for key, value in overview.items():
        print(f"{key}={value}")
    print(f"summary_csv={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
