#!/usr/bin/env python3
"""Validate hardware BFS event traces for the Loc-Gowalla dataset."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

from transport_key import (
    call_context,
    derive_hardware_contexts,
    logical_distribution_class,
    offset_feature,
    phase_class,
    physical_dpu_identity,
    same_source_across_group,
    sdk_api_kind,
    transport_key,
    transport_key_v6_full,
)


EXPECTED_LEVELS = 10
GRAPH_PATH = Path(__file__).parent / "data" / "loc-gowalla_edges.txt"


def round_up_to_8(value: int) -> int:
    return ((value + 7) // 8) * 8


@lru_cache(maxsize=1)
def graph_degrees() -> list[int]:
    with GRAPH_PATH.open() as stream:
        header_nodes, header_columns, header_edges = map(
            int, stream.readline().split()
        )
        num_nodes = ((max(header_nodes, header_columns) + 63) // 64) * 64
        degrees = [0] * num_nodes
        observed_edges = 0
        for line in stream:
            node, _ = map(int, line.split())
            degrees[node] += 1
            observed_edges += 1
    require(
        observed_edges == header_edges,
        f"graph edges={observed_edges}, header reports {header_edges}",
    )
    return degrees


def expected_metrics(nr_dpus: int) -> dict[str, object]:
    degrees = graph_degrees()
    num_nodes = len(degrees)
    require(nr_dpus >= 64, f"configured_dpus={nr_dpus} is below one rank")
    require(
        nr_dpus % 64 == 0,
        f"configured_dpus={nr_dpus} is not a whole number of 64-DPU ranks",
    )
    require(
        num_nodes % nr_dpus == 0,
        f"configured_dpus={nr_dpus} does not divide {num_nodes} nodes",
    )
    nodes_per_dpu = num_nodes // nr_dpus
    frontier_bytes = num_nodes // 64 * 8
    node_ptrs_logical = num_nodes * 4 + nr_dpus * 4
    neighbor_logical = sum(degrees) * 4
    neighbor_transfer = sum(
        round_up_to_8(
            sum(degrees[index : index + nodes_per_dpu]) * 4
        )
        for index in range(0, num_nodes, nodes_per_dpu)
    )
    logical_by_subop = {
        "node_ptrs": node_ptrs_logical,
        "neighbor_idxs": neighbor_logical,
        "node_level_init": num_nodes * 4,
        "visited_init": nr_dpus * frontier_bytes,
        "frontier_init": nr_dpus * frontier_bytes,
        "params_init": nr_dpus * 44,
        "frontier_broadcast": 9 * nr_dpus * frontier_bytes,
        "params_level": 9 * nr_dpus * 44,
        "frontier_result": 10 * nr_dpus * frontier_bytes,
        "node_level_result": num_nodes * 4,
    }
    transfer_by_subop = dict(logical_by_subop)
    transfer_by_subop.update(
        {
            "node_ptrs": nr_dpus * round_up_to_8(
                (nodes_per_dpu + 1) * 4
            ),
            "neighbor_idxs": neighbor_transfer,
            "params_init": nr_dpus * 48,
            "params_level": 9 * nr_dpus * 48,
        }
    )
    subop_counts = {
        "node_ptrs": nr_dpus,
        "neighbor_idxs": nr_dpus,
        "node_level_init": nr_dpus,
        "visited_init": nr_dpus,
        "frontier_init": nr_dpus,
        "params_init": nr_dpus,
        "frontier_broadcast": 9 * nr_dpus,
        "params_level": 9 * nr_dpus,
        "frontier_result": 10 * nr_dpus,
        "node_level_result": nr_dpus,
        "bfs_level": EXPECTED_LEVELS,
    }
    h2d_subops = {
        "node_ptrs",
        "neighbor_idxs",
        "node_level_init",
        "visited_init",
        "frontier_init",
        "params_init",
        "frontier_broadcast",
        "params_level",
    }
    d2h_subops = {"frontier_result", "node_level_result"}
    return {
        "actual_ranks": nr_dpus // 64,
        "events": 35 * nr_dpus + 13,
        "copy_to": 24 * nr_dpus,
        "copy_from": 11 * nr_dpus,
        "h2d_logical_bytes": sum(
            logical_by_subop[subop] for subop in h2d_subops
        ),
        "h2d_transfer_bytes": sum(
            transfer_by_subop[subop] for subop in h2d_subops
        ),
        "d2h_logical_bytes": sum(
            logical_by_subop[subop] for subop in d2h_subops
        ),
        "d2h_transfer_bytes": sum(
            transfer_by_subop[subop] for subop in d2h_subops
        ),
        "subop_counts": subop_counts,
        "logical_bytes_by_subop": logical_by_subop,
        "transfer_bytes_by_subop": transfer_by_subop,
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


def validate(
    path: Path,
    allow_legacy_transport_key: bool = False,
) -> dict[str, object]:
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
    expected = expected_metrics(nr_dpus)
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
    hardware_contexts = derive_hardware_contexts(rows)
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
    for row, expected_context in zip(rows, hardware_contexts):
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
            row["sdk_api_kind"] == sdk_api_kind(row["op"]),
            f"event {row['event_id']} has invalid "
            f"sdk_api_kind={row['sdk_api_kind']}",
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
            row["phase_class"] == phase_class(row["op"], row["subop"]),
            f"event {row['event_id']} has invalid phase_class",
        )
        for field, expected_value in expected_context.items():
            require(
                row[field] == expected_value,
                f"event {row['event_id']} has invalid {field}=<"
                f"{row[field]}>, expected <{expected_value}>",
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
            row["physical_dpu_identity"] == physical_dpu_identity(row),
            f"event {row['event_id']} has invalid physical_dpu_identity",
        )
        valid_transport_keys = {transport_key(row)}
        if allow_legacy_transport_key:
            valid_transport_keys.add(transport_key_v6_full(row))
        require(
            row["transport_key"] in valid_transport_keys,
            f"event {row['event_id']} has invalid transport_key",
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
        "sdk_api_kind",
        "logical_distribution_class",
        "target_space",
        "transfer_bytes_per_dpu",
        "active_dpus",
        "active_ranks",
        "active_dpus_per_rank",
        "global_dpu_id",
        "rank_ordinal",
        "dpu_id_in_rank",
        "sdk_physical_rank_id",
        "sdk_slice_id",
        "sdk_member_id",
        "physical_dpu_identity",
        "same_source_across_group",
        "phase_class",
        "target_symbol",
        "offset_bytes",
        "logical_bytes",
        "transfer_bytes",
        "host_buffer_address",
        "host_buffer_page_offset",
        "host_buffer_reuse_class",
        "previous_sdk_op",
        "previous_sdk_direction",
        "previous_sdk_transfer_bytes",
        "previous_sdk_topology_relation",
        "ns_since_previous_sdk_event",
        "previous_dpu_direction",
        "previous_dpu_transfer_bytes",
        "previous_dpu_target_relation",
        "launches_since_previous_dpu_transfer",
        "target_region_reuse_class",
        "transport_key",
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
        all(int(row["sdk_physical_rank_id"]) >= 0 for row in copy_rows),
        "copy event has an invalid sdk_physical_rank_id",
    )
    require(
        all(0 <= int(row["sdk_slice_id"]) < 8 for row in copy_rows),
        "copy event sdk_slice_id is outside 0..7",
    )
    require(
        all(0 <= int(row["sdk_member_id"]) < 8 for row in copy_rows),
        "copy event sdk_member_id is outside 0..7",
    )
    require(
        all(
            row["previous_sdk_topology_relation"]
            in {
                "NONE",
                "COLLECTION",
                "SAME_DPU",
                "SAME_MUX_PAIR",
                "SAME_SLICE",
                "SAME_RANK",
                "OTHER_RANK",
            }
            for row in copy_rows
        ),
        "copy event has an invalid previous SDK topology relation",
    )
    require(
        all(row["target_space"] == "MRAM" for row in copy_rows),
        "copy event target_space differs from MRAM",
    )
    require(
        all(row["sdk_api_kind"] == "SINGLE_COPY" for row in copy_rows),
        "copy event sdk_api_kind differs from SINGLE_COPY",
    )
    require(
        all(row["active_dpus"] == "1" for row in copy_rows),
        "copy event active_dpus differs from 1",
    )
    require(
        all(row["active_ranks"] == "1" for row in copy_rows),
        "copy event active_ranks differs from 1",
    )
    require(
        all(row["active_dpus_per_rank"] == "1" for row in copy_rows),
        "copy event active_dpus_per_rank differs from 1",
    )
    require(
        all(
            row["phase_class"] in {"INIT", "ITERATIVE", "FINALIZE"}
            for row in copy_rows
        ),
        "copy event phase_class is outside the supported BFS phases",
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
        require(
            int(row["transfer_bytes_per_dpu"]) == transfer,
            f"event {row['event_id']} has inconsistent transfer_bytes_per_dpu",
        )
        require(transfer % 8 == 0, f"event {row['event_id']} is not 8-byte aligned")
        require(
            logical <= transfer < logical + 8,
            f"event {row['event_id']} has invalid logical/transfer byte sizes",
        )

    dpu_topology = {}
    for row in copy_rows:
        dpu_id = int(row["global_dpu_id"])
        topology = (
            int(row["rank_ordinal"]),
            int(row["dpu_id_in_rank"]),
            int(row["sdk_physical_rank_id"]),
            int(row["sdk_slice_id"]),
            int(row["sdk_member_id"]),
        )
        require(
            dpu_id not in dpu_topology or dpu_topology[dpu_id] == topology,
            f"global_dpu_id={dpu_id} has inconsistent topology",
        )
        dpu_topology[dpu_id] = topology
    require(
        set(dpu_topology) == set(range(nr_dpus)),
        "copy events do not cover every configured DPU",
    )
    rank_members = Counter(topology[0] for topology in dpu_topology.values())
    require(
        rank_members == Counter({rank: 64 for rank in range(rank_count)}),
        f"rank membership={dict(rank_members)}, expected 64 DPUs per rank",
    )
    for rank in range(rank_count):
        dpu_ids = {
            dpu_id_in_rank
            for rank_ordinal, dpu_id_in_rank, _, _, _ in dpu_topology.values()
            if rank_ordinal == rank
        }
        require(
            dpu_ids == set(range(64)),
            f"rank {rank} DPU IDs differ from 0..63",
        )
        sdk_pairs = {
            (slice_id, member_id)
            for rank_ordinal, _, _, slice_id, member_id in dpu_topology.values()
            if rank_ordinal == rank
        }
        require(
            len(sdk_pairs) == 64,
            f"rank {rank} SDK slice/member pairs are not unique",
        )
    physical_dpu_identities = {
        (
            physical_rank_id,
            slice_id,
            member_id,
        )
        for _, _, physical_rank_id, slice_id, member_id
        in dpu_topology.values()
    }
    require(
        len(physical_dpu_identities) == nr_dpus,
        "physical rank/slice/member tuples are not unique",
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
    parser.add_argument("--allow-legacy-transport-key", action="store_true")
    args = parser.parse_args()

    summaries = []
    failed = False
    for path in args.traces:
        try:
            summary = validate(path, args.allow_legacy_transport_key)
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
