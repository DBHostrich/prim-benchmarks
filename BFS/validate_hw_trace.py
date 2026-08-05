#!/usr/bin/env python3
"""Validate hardware BFS event traces for the Loc-Gowalla dataset."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter
from pathlib import Path

from measurement_label import (
    api_type,
    call_context,
    logical_distribution_class,
    measurement_label,
    offset_feature,
    same_source_across_group,
)


EXPECTED_LEVELS = 10
EXPECTED = {
    256: {
        "actual_ranks": 4,
        "events": 8_973,
        "copy_to": 6_144,
        "copy_from": 2_816,
        "h2d_logical_bytes": 78_495_160,
        "h2d_transfer_bytes": 78_506_896,
        "d2h_logical_bytes": 63_700_992,
        "d2h_transfer_bytes": 63_700_992,
        "subop_counts": {
            "node_ptrs": 256,
            "neighbor_idxs": 256,
            "node_level_init": 256,
            "visited_init": 256,
            "frontier_init": 256,
            "params_init": 256,
            "frontier_broadcast": 2_304,
            "params_level": 2_304,
            "frontier_result": 2_560,
            "node_level_result": 256,
            "bfs_level": 10,
        },
        "logical_bytes_by_subop": {
            "node_ptrs": 787_456,
            "neighbor_idxs": 7_602_616,
            "node_level_init": 786_432,
            "visited_init": 6_291_456,
            "frontier_init": 6_291_456,
            "params_init": 11_264,
            "frontier_broadcast": 56_623_104,
            "params_level": 101_376,
            "frontier_result": 62_914_560,
            "node_level_result": 786_432,
        },
        "transfer_bytes_by_subop": {
            "node_ptrs": 788_480,
            "neighbor_idxs": 7_603_088,
            "node_level_init": 786_432,
            "visited_init": 6_291_456,
            "frontier_init": 6_291_456,
            "params_init": 12_288,
            "frontier_broadcast": 56_623_104,
            "params_level": 110_592,
            "frontier_result": 62_914_560,
            "node_level_result": 786_432,
        },
    },
    512: {
        "actual_ranks": 8,
        "events": 17_933,
        "copy_to": 12_288,
        "copy_from": 5_632,
        "h2d_logical_bytes": 147_814_840,
        "h2d_transfer_bytes": 147_838_376,
        "d2h_logical_bytes": 126_615_552,
        "d2h_transfer_bytes": 126_615_552,
        "subop_counts": {
            "node_ptrs": 512,
            "neighbor_idxs": 512,
            "node_level_init": 512,
            "visited_init": 512,
            "frontier_init": 512,
            "params_init": 512,
            "frontier_broadcast": 4_608,
            "params_level": 4_608,
            "frontier_result": 5_120,
            "node_level_result": 512,
            "bfs_level": 10,
        },
        "logical_bytes_by_subop": {
            "node_ptrs": 788_480,
            "neighbor_idxs": 7_602_616,
            "node_level_init": 786_432,
            "visited_init": 12_582_912,
            "frontier_init": 12_582_912,
            "params_init": 22_528,
            "frontier_broadcast": 113_246_208,
            "params_level": 202_752,
            "frontier_result": 125_829_120,
            "node_level_result": 786_432,
        },
        "transfer_bytes_by_subop": {
            "node_ptrs": 790_528,
            "neighbor_idxs": 7_603_624,
            "node_level_init": 786_432,
            "visited_init": 12_582_912,
            "frontier_init": 12_582_912,
            "params_init": 24_576,
            "frontier_broadcast": 113_246_208,
            "params_level": 221_184,
            "frontier_result": 125_829_120,
            "node_level_result": 786_432,
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


def expected_sequence(nr_dpus: int) -> list[tuple[str, str, str, str]]:
    sequence: list[tuple[str, str, str, str]] = [
        ("dpu_alloc", "", "", ""),
        ("dpu_load", "", "", ""),
    ]
    initial_subops = (
        "node_ptrs",
        "neighbor_idxs",
        "node_level_init",
        "visited_init",
        "frontier_init",
        "params_init",
    )
    for dpu_id in range(nr_dpus):
        for subop in initial_subops:
            sequence.append(("dpu_copy_to", subop, "", str(dpu_id)))

    for level in range(1, EXPECTED_LEVELS + 1):
        sequence.append(("dpu_launch", "bfs_level", str(level), ""))
        for dpu_id in range(nr_dpus):
            sequence.append(
                ("dpu_copy_from", "frontier_result", str(level), str(dpu_id))
            )
        if level < EXPECTED_LEVELS:
            next_level = str(level + 1)
            for dpu_id in range(nr_dpus):
                sequence.append(
                    ("dpu_copy_to", "frontier_broadcast", next_level, str(dpu_id))
                )
                sequence.append(
                    ("dpu_copy_to", "params_level", next_level, str(dpu_id))
                )

    for dpu_id in range(nr_dpus):
        sequence.append(("dpu_copy_from", "node_level_result", "", str(dpu_id)))
    sequence.append(("dpu_free", "", "", ""))
    return sequence


def validate(path: Path) -> dict[str, object]:
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
    rank_count = actual_ranks.pop()
    require(
        rank_count == expected["actual_ranks"],
        f"actual_ranks={rank_count}, expected {expected['actual_ranks']}",
    )
    require(
        len(rows) == expected["events"],
        f"event rows={len(rows)}, expected {expected['events']}",
    )

    event_ids = [int(row["event_id"]) for row in rows]
    require(event_ids == list(range(len(rows))), "event_id is not contiguous from zero")
    for row in rows:
        start = int(row["host_start_ns"])
        end = int(row["host_end_ns"])
        elapsed = int(row["measured_ns"])
        require(end >= start, f"event {row['event_id']} has a negative duration")
        require(
            elapsed == end - start,
            f"event {row['event_id']} measured_ns != end-start",
        )
        require(elapsed > 0, f"event {row['event_id']} has a non-positive duration")

    process_states = {row["process_state"] for row in rows}
    warmup_counts = {row["pretrace_warmup_runs"] for row in rows}
    host_numa_nodes = {row["host_numa_node"] for row in rows}
    require(len(process_states) == 1 and "" not in process_states,
            f"invalid process_state values: {process_states}")
    require(len(warmup_counts) == 1 and "" not in warmup_counts,
            f"invalid pretrace_warmup_runs values: {warmup_counts}")
    require(len(host_numa_nodes) == 1 and "" not in host_numa_nodes,
            f"invalid host_numa_node values: {host_numa_nodes}")

    op_call_counts = Counter()
    dpu_op_call_counts = Counter()
    for row in rows:
        expected_op_index = op_call_counts[row["op"]]
        require(
            int(row["op_call_index"]) == expected_op_index,
            f"event {row['event_id']} has op_call_index={row['op_call_index']}, "
            f"expected {expected_op_index}",
        )
        op_call_counts[row["op"]] += 1

        has_dpu_call_index = row["op"] in {"dpu_copy_to", "dpu_copy_from"}
        if has_dpu_call_index:
            key = (row["op"], row["global_dpu_id"])
            expected_dpu_index = dpu_op_call_counts[key]
            require(
                int(row["dpu_op_call_index"]) == expected_dpu_index,
                f"event {row['event_id']} has dpu_op_call_index="
                f"{row['dpu_op_call_index']}, expected {expected_dpu_index}",
            )
            dpu_op_call_counts[key] += 1
        else:
            require(
                row["dpu_op_call_index"] == "",
                f"event {row['event_id']} has unexpected dpu_op_call_index",
            )

        require(
            row["api_type"] == api_type(row["op"]),
            f"event {row['event_id']} has invalid api_type={row['api_type']}",
        )
        require(
            row["logical_distribution_class"]
            == logical_distribution_class(row["op"], row["subop"]),
            f"event {row['event_id']} has invalid logical_distribution_class",
        )
        require(
            row["same_source_across_group"]
            == same_source_across_group(row["op"], row["subop"]),
            f"event {row['event_id']} has invalid same_source_across_group",
        )
        require(
            row["offset_feature"] == offset_feature(row),
            f"event {row['event_id']} has invalid offset_feature",
        )
        require(
            row["call_context"] == call_context(row),
            f"event {row['event_id']} has invalid call_context",
        )
        require(
            row["measurement_label"] == measurement_label(row),
            f"event {row['event_id']} has invalid measurement_label",
        )

    ops = Counter(row["op"] for row in rows)
    require(ops["dpu_alloc"] == 1, f"alloc={ops['dpu_alloc']}, expected 1")
    require(ops["dpu_load"] == 1, f"load={ops['dpu_load']}, expected 1")
    require(
        ops["dpu_copy_to"] == expected["copy_to"],
        f"copy_to={ops['dpu_copy_to']}, expected {expected['copy_to']}",
    )
    require(
        ops["dpu_copy_from"] == expected["copy_from"],
        f"copy_from={ops['dpu_copy_from']}, expected {expected['copy_from']}",
    )
    require(
        ops["dpu_launch"] == EXPECTED_LEVELS,
        f"launch={ops['dpu_launch']}, expected {EXPECTED_LEVELS}",
    )
    require(ops["dpu_free"] == 1, f"free={ops['dpu_free']}, expected 1")

    actual_sequence = [
        (row["op"], row["subop"], row["bfs_level"], row["global_dpu_id"])
        for row in rows
    ]
    wanted_sequence = expected_sequence(nr_dpus)
    require(
        len(actual_sequence) == len(wanted_sequence),
        "internal sequence length differs from expected event count",
    )
    for event_id, (actual, wanted) in enumerate(zip(actual_sequence, wanted_sequence)):
        require(
            actual == wanted,
            f"event {event_id} semantic tuple={actual}, expected {wanted}",
        )

    subops = Counter(
        row["subop"]
        for row in rows
        if row["op"] in {"dpu_copy_to", "dpu_copy_from", "dpu_launch"}
    )
    require(
        subops == Counter(expected["subop_counts"]),
        f"subop counts={dict(subops)}, expected {expected['subop_counts']}",
    )

    collection_rows = [
        row
        for row in rows
        if row["op"] in {"dpu_alloc", "dpu_load", "dpu_launch", "dpu_free"}
    ]
    collection_empty_fields = (
        "global_dpu_id",
        "rank_ordinal",
        "dpu_id_in_rank",
        "target_space",
        "target_symbol",
        "offset_bytes",
        "logical_bytes",
        "transfer_bytes",
    )
    require(
        all(
            row[field] == ""
            for row in collection_rows
            for field in collection_empty_fields
        ),
        "collection event contains DPU topology, target, or byte fields",
    )

    launch_rows = [row for row in rows if row["op"] == "dpu_launch"]
    require(
        [int(row["bfs_level"]) for row in launch_rows]
        == list(range(1, EXPECTED_LEVELS + 1)),
        "launch bfs_level values differ from 1..10",
    )
    require(
        all(row["direction"] == "" for row in launch_rows),
        "launch event has a direction",
    )

    copy_rows = [row for row in rows if row["op"].startswith("dpu_copy_")]
    require(
        all(row["global_dpu_id"] != "" for row in copy_rows),
        "copy event has an empty global_dpu_id",
    )
    require(
        all(0 <= int(row["rank_ordinal"]) < rank_count for row in copy_rows),
        "copy event rank_ordinal is outside actual_ranks",
    )
    require(
        all(row["target_space"] == "MRAM" for row in copy_rows),
        "copy event target_space differs from MRAM",
    )
    require(
        all(
            row["target_symbol"] == "DPU_MRAM_HEAP_POINTER_NAME"
            for row in copy_rows
        ),
        "copy event target_symbol differs from DPU_MRAM_HEAP_POINTER_NAME",
    )
    require(
        all(
            row["direction"]
            == ("TO_DPU" if row["op"] == "dpu_copy_to" else "FROM_DPU")
            for row in copy_rows
        ),
        "copy event direction differs from op",
    )
    for row in copy_rows:
        logical = int(row["logical_bytes"])
        transfer = int(row["transfer_bytes"])
        require(transfer % 8 == 0, f"event {row['event_id']} is not 8-byte aligned")
        require(
            logical <= transfer < logical + 8,
            f"event {row['event_id']} has invalid logical/transfer byte sizes",
        )

    dpu_topology = {}
    for row in copy_rows:
        dpu_id = int(row["global_dpu_id"])
        topology = (int(row["rank_ordinal"]), int(row["dpu_id_in_rank"]))
        require(
            dpu_id not in dpu_topology or dpu_topology[dpu_id] == topology,
            f"global_dpu_id={dpu_id} has inconsistent topology",
        )
        dpu_topology[dpu_id] = topology
    require(
        set(dpu_topology) == set(range(nr_dpus)),
        "copy events do not cover every configured DPU",
    )
    rank_members = Counter(rank for rank, _ in dpu_topology.values())
    require(
        rank_members == Counter({rank: 64 for rank in range(rank_count)}),
        f"rank membership={dict(rank_members)}, expected 64 DPUs per rank",
    )
    for rank in range(rank_count):
        dpu_ids = {
            dpu_id_in_rank
            for rank_ordinal, dpu_id_in_rank in dpu_topology.values()
            if rank_ordinal == rank
        }
        require(
            dpu_ids == set(range(64)),
            f"rank {rank} DPU IDs differ from 0..63",
        )

    logical_by_subop = Counter()
    transfer_by_subop = Counter()
    for row in copy_rows:
        logical_by_subop[row["subop"]] += int(row["logical_bytes"])
        transfer_by_subop[row["subop"]] += int(row["transfer_bytes"])
    require(
        logical_by_subop == Counter(expected["logical_bytes_by_subop"]),
        f"logical bytes by subop={dict(logical_by_subop)}",
    )
    require(
        transfer_by_subop == Counter(expected["transfer_bytes_by_subop"]),
        f"transfer bytes by subop={dict(transfer_by_subop)}",
    )

    h2d_rows = [row for row in rows if row["op"] == "dpu_copy_to"]
    d2h_rows = [row for row in rows if row["op"] == "dpu_copy_from"]
    h2d_logical = sum(int(row["logical_bytes"]) for row in h2d_rows)
    h2d_transfer = sum(int(row["transfer_bytes"]) for row in h2d_rows)
    d2h_logical = sum(int(row["logical_bytes"]) for row in d2h_rows)
    d2h_transfer = sum(int(row["transfer_bytes"]) for row in d2h_rows)
    require(
        h2d_logical == expected["h2d_logical_bytes"],
        f"H2D logical bytes={h2d_logical}, expected {expected['h2d_logical_bytes']}",
    )
    require(
        h2d_transfer == expected["h2d_transfer_bytes"],
        f"H2D transfer bytes={h2d_transfer}, expected {expected['h2d_transfer_bytes']}",
    )
    require(
        d2h_logical == expected["d2h_logical_bytes"],
        f"D2H logical bytes={d2h_logical}, expected {expected['d2h_logical_bytes']}",
    )
    require(
        d2h_transfer == expected["d2h_transfer_bytes"],
        f"D2H transfer bytes={d2h_transfer}, expected {expected['d2h_transfer_bytes']}",
    )

    launch_by_level = {
        int(row["bfs_level"]): int(row["measured_ns"]) for row in launch_rows
    }
    measured = {
        "alloc_measured_ns": sum(
            int(row["measured_ns"]) for row in rows if row["op"] == "dpu_alloc"
        ),
        "load_measured_ns": sum(
            int(row["measured_ns"]) for row in rows if row["op"] == "dpu_load"
        ),
        "h2d_measured_ns": sum(int(row["measured_ns"]) for row in h2d_rows),
        "launch_measured_ns": sum(
            int(row["measured_ns"]) for row in launch_rows
        ),
        "d2h_measured_ns": sum(int(row["measured_ns"]) for row in d2h_rows),
        "free_measured_ns": sum(
            int(row["measured_ns"]) for row in rows if row["op"] == "dpu_free"
        ),
    }
    return {
        "run_id": run_ids.pop(),
        "repeat_id": repeats.pop(),
        "configured_dpus": nr_dpus,
        "num_tasklets": tasklets.pop(),
        "actual_ranks": rank_count,
        "events": len(rows),
        "h2d_logical_bytes": h2d_logical,
        "h2d_transfer_bytes": h2d_transfer,
        "d2h_logical_bytes": d2h_logical,
        "d2h_transfer_bytes": d2h_transfer,
        "host_clock_measured_ns": sum(int(row["measured_ns"]) for row in rows),
        "launch_by_level": launch_by_level,
        **measured,
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
                f"events={summary['events']} "
                f"H2D={summary['h2d_transfer_bytes']} "
                f"D2H={summary['d2h_transfer_bytes']} "
                f"levels={EXPECTED_LEVELS}"
            )

    if failed:
        return 1
    if summaries:
        for key in (
            "alloc_measured_ns",
            "load_measured_ns",
            "h2d_measured_ns",
            "launch_measured_ns",
            "d2h_measured_ns",
            "free_measured_ns",
            "host_clock_measured_ns",
        ):
            values = [int(summary[key]) for summary in summaries]
            print(f"{key}_median={statistics.median(values):.0f}")
        for level in range(1, EXPECTED_LEVELS + 1):
            values = [
                int(summary["launch_by_level"][level]) for summary in summaries
            ]
            print(
                f"launch_level_{level:02d}_measured_ns_median="
                f"{statistics.median(values):.0f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
