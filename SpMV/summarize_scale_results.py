#!/usr/bin/env python3
"""Build one stability row per SpMV scale configuration."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open() as stream:
        for line in stream:
            name, separator, value = line.rstrip("\n").partition("=")
            if separator:
                values[name] = value
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for result_dir in sorted(args.result_root.glob("SpMV_*dpu_*tl")):
        config = read_key_values(result_dir / "config.txt")
        analysis = read_key_values(result_dir / "transport_key_analysis.log")
        validation_lines = (result_dir / "validation.log").read_text().splitlines()
        valid_traces = sum(line.startswith("PASS ") for line in validation_lines)
        expected_traces = int(config["N_REPS"])
        unstable_groups = int(analysis["unstable_groups"])
        insufficient_groups = int(analysis["insufficient_groups"])
        stable_event_pct = float(analysis["stable_event_pct"])
        if valid_traces != expected_traces:
            status = "INVALID"
        elif insufficient_groups:
            status = "INSUFFICIENT"
        elif unstable_groups == 0 and stable_event_pct == 100.0:
            status = "STABLE"
        else:
            status = "MIXED"
        rows.append(
            {
                "config": result_dir.name,
                "configured_dpus": config["NR_DPUS"],
                "allocated_ranks": str(int(config["NR_DPUS"]) // 64),
                "expected_traces": str(expected_traces),
                "valid_traces": str(valid_traces),
                "transfer_rows": analysis["transfer_rows"],
                "transport_key_groups": analysis["transport_key_groups"],
                "stable_groups": analysis["stable_groups"],
                "unstable_groups": analysis["unstable_groups"],
                "insufficient_groups": analysis["insufficient_groups"],
                "stable_transport_key_pct": analysis[
                    "stable_transport_key_pct"
                ],
                "stable_event_pct": analysis["stable_event_pct"],
                "baseline_v2_stable_transport_key_pct": analysis[
                    "baseline_v2_stable_transport_key_pct"
                ],
                "baseline_v2_stable_event_pct": analysis[
                    "baseline_v2_stable_event_pct"
                ],
                "status": status,
            }
        )
    if not rows:
        raise SystemExit(f"no SpMV configuration directories under {args.result_root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['config']} status={row['status']} "
            f"stable_keys={row['stable_transport_key_pct']}% "
            f"stable_events={row['stable_event_pct']}%"
        )
    print(f"summary_csv={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
