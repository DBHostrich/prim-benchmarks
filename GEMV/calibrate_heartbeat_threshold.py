#!/usr/bin/env python3
"""Choose a sparse GEMV heartbeat threshold from an idle calibration trace."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


def nearest_rank_percentile(values: list[int], percentile: float) -> int:
    if not values:
        raise ValueError("heartbeat calibration contains zero samples")
    if percentile <= 0 or percentile > 100:
        raise ValueError("percentile must be in (0, 100]")
    ordered = sorted(values)
    index = max(0, math.ceil(percentile / 100.0 * len(ordered)) - 1)
    return ordered[index]


def calibrate(
    path: Path,
    minimum_us: int = 200,
    percentile: float = 99.0,
) -> dict[str, object]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    lateness_ns = [int(row["lateness_ns"]) for row in rows]
    if len(lateness_ns) < 100:
        raise ValueError(
            f"heartbeat calibration needs at least 100 samples, got {len(lateness_ns)}"
        )
    percentile_ns = nearest_rank_percentile(lateness_ns, percentile)
    chosen_us = max(minimum_us, math.ceil(percentile_ns / 1000))
    return {
        "sample_count": len(lateness_ns),
        "median_lateness_us": f"{statistics.median(lateness_ns) / 1000:.3f}",
        "p90_lateness_us": f"{nearest_rank_percentile(lateness_ns, 90) / 1000:.3f}",
        "calibration_percentile": f"{percentile:.3f}",
        "calibration_percentile_lateness_us": f"{percentile_ns / 1000:.3f}",
        "maximum_lateness_us": f"{max(lateness_ns) / 1000:.3f}",
        "minimum_threshold_us": minimum_us,
        "chosen_threshold_us": chosen_us,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration_csv", type=Path)
    parser.add_argument("--minimum-us", type=int, default=200)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.minimum_us < 1:
        parser.error("minimum threshold must be positive")
    try:
        summary = calibrate(
            args.calibration_csv,
            minimum_us=args.minimum_us,
            percentile=args.percentile,
        )
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary))
            writer.writeheader()
            writer.writerow(summary)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(summary["chosen_threshold_us"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
