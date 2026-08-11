#!/usr/bin/env python3
"""Create one stability row per SCAN DPU scale."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def summarize(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_scale: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        scale = int(row["allocated_dpus"])
        by_scale.setdefault(scale, []).append(row)
    result: list[dict[str, object]] = []
    for scale, scale_rows in sorted(by_scale.items()):
        status = Counter(row["status"] for row in scale_rows)
        eligible = status["stable"] + status["unstable"]
        result.append(
            {
                "allocated_dpus": scale,
                "allocated_ranks": scale_rows[0]["allocated_ranks"],
                "dpu_sysfs_rank_ids": scale_rows[0]["dpu_sysfs_rank_ids"],
                "dpu_channel_ids": scale_rows[0]["dpu_channel_ids"],
                "key_groups": len(scale_rows),
                "stable_groups": status["stable"],
                "unstable_groups": status["unstable"],
                "insufficient_groups": status["insufficient"],
                "stable_key_pct": (
                    "0.000"
                    if eligible == 0
                    else f"{100.0 * status['stable'] / eligible:.3f}"
                ),
                "max_spread_pct": f"{max(float(row['p90_p10_spread_pct']) for row in scale_rows):.3f}",
                "max_cv_pct": f"{max(float(row['cv_pct']) for row in scale_rows):.3f}",
                "status": (
                    "insufficient"
                    if status["insufficient"]
                    else "stable"
                    if status["unstable"] == 0
                    else "unstable"
                ),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = summarize(args.summary)
    if not rows:
        raise SystemExit("transport-key summary contains zero rows")
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

