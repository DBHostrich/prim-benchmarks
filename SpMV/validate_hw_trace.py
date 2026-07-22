#!/usr/bin/env python3
"""Validate hardware SpMV event traces for the bcsstk30_base dataset."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter
from pathlib import Path


EXPECTED = {
    256: {
        "events": 1273,
        "copy_to": 1018,
        "copy_from": 254,
        "h2d_bytes": 37_800_320,
        "d2h_bytes": 115_696,
        "subops": {
            "params": 256,
            "row_ptrs": 254,
            "nonzeros": 254,
            "input_vector": 254,
            "output_vector": 254,
            "sync": 1,
        },
    },
    512: {
        "events": 2509,
        "copy_to": 2009,
        "copy_from": 499,
        "h2d_bytes": 66_153_944,
        "d2h_bytes": 115_696,
        "subops": {
            "params": 512,
            "row_ptrs": 499,
            "nonzeros": 499,
            "input_vector": 499,
            "output_vector": 499,
            "sync": 1,
        },
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_trace(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(bool(rows), "contains no event rows")
    return rows


def validate(path: Path) -> dict[str, int | str]:
    rows = read_trace(path)
    configured = {int(row["configured_dpus"]) for row in rows}
    tasklets = {int(row["num_tasklets"]) for row in rows}
    run_ids = {row["run_id"] for row in rows}
    repeats = {int(row["repeat_id"]) for row in rows}
    actual_ranks = {int(row["actual_ranks"]) for row in rows}

    require(len(configured) == 1, f"multiple configured_dpus values: {configured}")
    require(len(tasklets) == 1, f"multiple num_tasklets values: {tasklets}")
    require(len(run_ids) == 1, f"multiple run_id values: {run_ids}")
    require(len(repeats) == 1, f"multiple repeat_id values: {repeats}")
    require(len(actual_ranks) == 1, f"multiple actual_ranks values: {actual_ranks}")

    nr_dpus = configured.pop()
    require(nr_dpus in EXPECTED, f"unsupported configured_dpus={nr_dpus}")
    expected = EXPECTED[nr_dpus]
    require(len(rows) == expected["events"],
            f"event rows={len(rows)}, expected {expected['events']}")

    event_ids = [int(row["event_id"]) for row in rows]
    require(event_ids == list(range(len(rows))), "event_id is not contiguous from zero")
    for row in rows:
        start = int(row["host_start_ns"])
        end = int(row["host_end_ns"])
        elapsed = int(row["measured_ns"])
        require(end >= start, f"event {row['event_id']} has a negative duration")
        require(elapsed == end - start,
                f"event {row['event_id']} measured_ns != end-start")

    ops = Counter(row["op"] for row in rows)
    require(ops["dpu_copy_to"] == expected["copy_to"],
            f"copy_to={ops['dpu_copy_to']}, expected {expected['copy_to']}")
    require(ops["dpu_copy_from"] == expected["copy_from"],
            f"copy_from={ops['dpu_copy_from']}, expected {expected['copy_from']}")
    require(ops["dpu_launch"] == 1, f"launch={ops['dpu_launch']}, expected 1")

    subops = Counter(row["subop"] for row in rows)
    require(subops == Counter(expected["subops"]),
            f"subop counts={dict(subops)}, expected {expected['subops']}")

    h2d = sum(int(row["transfer_bytes"]) for row in rows if row["op"] == "dpu_copy_to")
    d2h = sum(int(row["transfer_bytes"]) for row in rows if row["op"] == "dpu_copy_from")
    require(h2d == expected["h2d_bytes"],
            f"H2D bytes={h2d}, expected {expected['h2d_bytes']}")
    require(d2h == expected["d2h_bytes"],
            f"D2H bytes={d2h}, expected {expected['d2h_bytes']}")

    rank_count = actual_ranks.pop()
    copy_rows = [row for row in rows if row["op"].startswith("dpu_copy_")]
    require(all(row["global_dpu_id"] != "" for row in copy_rows),
            "copy event has an empty global_dpu_id")
    require(all(0 <= int(row["rank_ordinal"]) < rank_count for row in copy_rows),
            "copy event rank_ordinal is outside actual_ranks")

    return {
        "run_id": run_ids.pop(),
        "repeat_id": repeats.pop(),
        "configured_dpus": nr_dpus,
        "num_tasklets": tasklets.pop(),
        "actual_ranks": rank_count,
        "events": len(rows),
        "h2d_bytes": h2d,
        "d2h_bytes": d2h,
        "h2d_measured_ns": sum(
            int(row["measured_ns"]) for row in rows if row["op"] == "dpu_copy_to"
        ),
        "d2h_measured_ns": sum(
            int(row["measured_ns"]) for row in rows if row["op"] == "dpu_copy_from"
        ),
        "launch_measured_ns": sum(
            int(row["measured_ns"]) for row in rows if row["op"] == "dpu_launch"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    args = parser.parse_args()

    summaries = []
    failed = False
    for path in args.traces:
        try:
            summary = validate(path)
        except (OSError, KeyError, TypeError, ValueError) as error:
            print(f"FAIL {path}: {error}", file=sys.stderr)
            failed = True
        else:
            summaries.append(summary)
            print(
                f"PASS {path}: DPU={summary['configured_dpus']} "
                f"TL={summary['num_tasklets']} ranks={summary['actual_ranks']} "
                f"events={summary['events']} H2D={summary['h2d_bytes']} "
                f"D2H={summary['d2h_bytes']}"
            )

    if failed:
        return 1
    if summaries:
        for key in ("h2d_measured_ns", "launch_measured_ns", "d2h_measured_ns"):
            values = [int(summary[key]) for summary in summaries]
            print(f"{key}_median={statistics.median(values):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
