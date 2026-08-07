#!/usr/bin/env python3
"""Select nested, channel-round-robin DPU rank sets from a topology TSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


REQUIRED_FIELDS = {
    "rank_path",
    "sysfs_rank_id",
    "numa",
    "channel",
    "dpus_per_rank",
    "status",
}


def parse_int_set(value: str) -> set[int]:
    return {int(token) for token in value.split(",") if token.strip()}


def selected_rows(
    path: Path,
    numa_node: int,
    excluded_sysfs_ranks: set[int],
) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_FIELDS - fields
        if missing:
            raise ValueError(f"topology is missing fields: {sorted(missing)}")
        rows = [
            row
            for row in reader
            if int(row["numa"]) == numa_node
            and row["status"] == "ok"
            and int(row["dpus_per_rank"]) == 64
            and int(row["sysfs_rank_id"]) not in excluded_sysfs_ranks
        ]
    by_channel: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_channel[int(row["channel"])].append(row)
    for channel_rows in by_channel.values():
        channel_rows.sort(key=lambda row: int(row["sysfs_rank_id"]))

    ordered: list[dict[str, str]] = []
    depth = 0
    while True:
        added = False
        for channel in sorted(by_channel):
            channel_rows = by_channel[channel]
            if depth < len(channel_rows):
                ordered.append(channel_rows[depth])
                added = True
        if not added:
            break
        depth += 1
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("topology", type=Path)
    parser.add_argument("--numa-node", type=int, default=0)
    parser.add_argument("--exclude-sysfs-ranks", type=parse_int_set, default=set())
    parser.add_argument("--rank-count", required=True, type=int)
    parser.add_argument(
        "--field",
        choices=("rank_path", "sysfs_rank_id", "channel"),
        default="rank_path",
    )
    args = parser.parse_args()
    if args.rank_count < 1:
        parser.error("--rank-count must be positive")
    rows = selected_rows(
        args.topology, args.numa_node, args.exclude_sysfs_ranks
    )
    if len(rows) < args.rank_count:
        raise SystemExit(
            f"NUMA {args.numa_node} has {len(rows)} usable ranks after exclusions, "
            f"need {args.rank_count}"
        )
    print(",".join(row[args.field] for row in rows[: args.rank_count]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
