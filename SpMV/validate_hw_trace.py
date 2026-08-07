#!/usr/bin/env python3
"""Validate hardware SpMV event traces for the bcsstk30_base dataset."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter, defaultdict
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
)

MATRIX_PATH = Path(__file__).parent / "data" / "bcsstk30.mtx"


def round_up_to_8(value: int) -> int:
    return ((value + 7) // 8) * 8


@lru_cache(maxsize=1)
def matrix_shape_and_row_counts() -> tuple[int, int, list[int]]:
    with MATRIX_PATH.open() as stream:
        num_rows, num_cols, num_nonzeros = map(int, stream.readline().split())
        if num_rows % 2:
            num_rows += 1
        row_counts = [0] * num_rows
        observed = 0
        for line in stream:
            row, _ = map(int, line.split())
            row_counts[row - 1] += 1
            observed += 1
    require(observed == num_nonzeros,
            f"matrix entries={observed}, header reports {num_nonzeros}")
    return num_rows, num_cols, row_counts


def expected_metrics(nr_dpus: int) -> dict[str, object]:
    require(nr_dpus >= 64, f"configured_dpus={nr_dpus} is below one rank")
    require(nr_dpus % 64 == 0,
            f"configured_dpus={nr_dpus} is not rank aligned")
    num_rows, num_cols, row_counts = matrix_shape_and_row_counts()
    rows_per_dpu = ((num_rows - 1) // nr_dpus + 2) // 2 * 2
    partitions: list[tuple[int, int]] = []
    for dpu_id in range(nr_dpus):
        start = dpu_id * rows_per_dpu
        count = max(0, min(rows_per_dpu, num_rows - start))
        partitions.append((start, count))
    active = [(start, count) for start, count in partitions if count > 0]
    active_dpus = len(active)
    row_ptr_bytes = sum(round_up_to_8((count + 1) * 4) for _, count in active)
    nonzero_bytes = sum(
        round_up_to_8(sum(row_counts[start:start + count]) * 8)
        for start, count in active
    )
    input_bytes = active_dpus * round_up_to_8(num_cols * 4)
    params_bytes = nr_dpus * round_up_to_8(20)
    output_bytes = sum(round_up_to_8(count * 4) for _, count in active)
    return {
        "actual_ranks": nr_dpus // 64,
        "events": 2 + active_dpus * 4 + nr_dpus + 2,
        "copy_to": active_dpus * 3 + nr_dpus,
        "copy_from": active_dpus,
        "h2d_bytes": row_ptr_bytes + nonzero_bytes + input_bytes + params_bytes,
        "d2h_bytes": output_bytes,
        "partitions": partitions,
        "subops": {
            "params": nr_dpus,
            "row_ptrs": active_dpus,
            "nonzeros": active_dpus,
            "input_vector": active_dpus,
            "output_vector": active_dpus,
            "sync": 1,
        },
    }

EVENT_FIELDS = {
    "run_id",
    "repeat_id",
    "event_id",
    "configured_dpus",
    "actual_ranks",
    "num_tasklets",
    "op",
    "direction",
    "sdk_api_kind",
    "logical_distribution_class",
    "target_space",
    "transfer_bytes_per_dpu",
    "active_dpus",
    "active_ranks",
    "active_dpus_per_rank",
    "rank_ordinal",
    "dpu_id_in_rank",
    "sdk_physical_rank_id",
    "dpu_sysfs_rank_id",
    "dpu_rank_numa_node",
    "dpu_channel_id",
    "sdk_slice_id",
    "sdk_member_id",
    "dpu_ci_id",
    "dpu_member_id",
    "physical_dpu_identity",
    "cpu_dpu_numa_relation",
    "same_source_across_group",
    "phase_class",
    "subop",
    "global_dpu_id",
    "target_symbol",
    "offset_bytes",
    "offset_feature",
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
    "op_call_index",
    "dpu_op_call_index",
    "process_state",
    "pretrace_warmup_runs",
    "host_numa_node",
    "call_context",
    "transport_key",
    "host_start_ns",
    "host_end_ns",
    "measured_ns",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_int_set(value: str) -> set[int]:
    result: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError(f"invalid range: {token}")
            result.update(range(start, end + 1))
        else:
            result.add(int(token))
    if not result:
        raise argparse.ArgumentTypeError("expected a non-empty integer set")
    return result


def expected_cpu_dpu_numa_relation(row: dict[str, str]) -> str:
    host_numa = row["host_numa_node"]
    if host_numa == "unbound":
        return "UNBOUND"
    if not host_numa.isdigit():
        return "UNKNOWN"
    return "LOCAL" if host_numa == row["dpu_rank_numa_node"] else "REMOTE"


def read_trace(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        require(
            fields == EVENT_FIELDS,
            f"header fields differ: got {sorted(fields)}",
        )
        rows = list(reader)
    require(bool(rows), "contains no event rows")
    return rows


def expected_sequence(
    nr_dpus: int, nonempty_dpus: int
) -> list[tuple[str, str, str]]:
    sequence = [("dpu_alloc", "", ""), ("dpu_load", "", "")]
    for dpu_id in range(nr_dpus):
        if dpu_id < nonempty_dpus:
            sequence.extend(
                ("dpu_copy_to", subop, str(dpu_id))
                for subop in ("row_ptrs", "nonzeros", "input_vector")
            )
        sequence.append(("dpu_copy_to", "params", str(dpu_id)))
    sequence.append(("dpu_launch", "sync", ""))
    sequence.extend(
        ("dpu_copy_from", "output_vector", str(dpu_id))
        for dpu_id in range(nonempty_dpus)
    )
    sequence.append(("dpu_free", "", ""))
    return sequence


def validate(
    path: Path,
    expected_host_numa_node: int | None = None,
    expected_dpu_numa_node: int | None = None,
    expected_sysfs_ranks: set[int] | None = None,
) -> dict[str, int | str]:
    rows = read_trace(path)
    configured = {int(row["configured_dpus"]) for row in rows}
    tasklets = {int(row["num_tasklets"]) for row in rows}
    run_ids = {row["run_id"] for row in rows}
    repeats = {int(row["repeat_id"]) for row in rows}
    actual_ranks = {int(row["actual_ranks"]) for row in rows}
    process_states = {row["process_state"] for row in rows}
    pretrace_warmup_runs = {
        int(row["pretrace_warmup_runs"]) for row in rows
    }
    host_numa_nodes = {row["host_numa_node"] for row in rows}

    require(len(configured) == 1, f"multiple configured_dpus values: {configured}")
    require(len(tasklets) == 1, f"multiple num_tasklets values: {tasklets}")
    require(len(run_ids) == 1, f"multiple run_id values: {run_ids}")
    require(len(repeats) == 1, f"multiple repeat_id values: {repeats}")
    require(len(actual_ranks) == 1, f"multiple actual_ranks values: {actual_ranks}")
    require(
        len(process_states) == 1 and next(iter(process_states)),
        f"process_state metadata is invalid: {process_states}",
    )
    require(
        len(pretrace_warmup_runs) == 1,
        f"multiple pretrace_warmup_runs values: {pretrace_warmup_runs}",
    )
    require(
        len(host_numa_nodes) == 1 and next(iter(host_numa_nodes)),
        f"host_numa_node metadata is invalid: {host_numa_nodes}",
    )
    if expected_host_numa_node is not None:
        require(
            host_numa_nodes == {str(expected_host_numa_node)},
            f"host_numa_node values={host_numa_nodes}, expected "
            f"{expected_host_numa_node}",
        )

    nr_dpus = configured.pop()
    require(
        next(iter(tasklets)) in {1, 2, 4, 8, 16},
        f"unsupported num_tasklets={next(iter(tasklets))}",
    )
    expected = expected_metrics(nr_dpus)
    require(
        next(iter(actual_ranks)) == expected["actual_ranks"],
        f"actual_ranks={next(iter(actual_ranks))}, "
        f"expected {expected['actual_ranks']}",
    )
    require(len(rows) == expected["events"],
            f"event rows={len(rows)}, expected {expected['events']}")

    event_ids = [int(row["event_id"]) for row in rows]
    require(event_ids == list(range(len(rows))), "event_id is not contiguous from zero")
    hardware_contexts = derive_hardware_contexts(rows)
    op_call_counts: dict[str, int] = defaultdict(int)
    dpu_op_call_counts: dict[tuple[str, int], int] = defaultdict(int)
    for row, expected_context in zip(rows, hardware_contexts):
        event_id = row["event_id"]
        start = int(row["host_start_ns"])
        end = int(row["host_end_ns"])
        elapsed = int(row["measured_ns"])
        require(end >= start, f"event {event_id} has a negative duration")
        require(elapsed == end - start,
                f"event {event_id} measured_ns != end-start")
        require(elapsed > 0, f"event {event_id} measured_ns is zero")
        require(
            int(row["op_call_index"]) == op_call_counts[row["op"]],
            f"event {event_id} has invalid op_call_index",
        )
        op_call_counts[row["op"]] += 1
        if row["op"] in {"dpu_copy_to", "dpu_copy_from"}:
            global_dpu_id = int(row["global_dpu_id"])
            dpu_key = (row["op"], global_dpu_id)
            require(
                int(row["dpu_op_call_index"])
                == dpu_op_call_counts[dpu_key],
                f"event {event_id} has invalid dpu_op_call_index",
            )
            dpu_op_call_counts[dpu_key] += 1
        else:
            require(
                row["dpu_op_call_index"] == "",
                f"event {event_id} has unexpected dpu_op_call_index",
            )
        require(
            row["sdk_api_kind"] == sdk_api_kind(row["op"]),
            f"event {event_id} has invalid sdk_api_kind",
        )
        require(
            row["logical_distribution_class"]
            == logical_distribution_class(row["op"], row["subop"]),
            f"event {event_id} has invalid logical_distribution_class",
        )
        require(
            row["same_source_across_group"]
            == same_source_across_group(row["op"], row["subop"]),
            f"event {event_id} has invalid same_source_across_group",
        )
        require(
            row["phase_class"] == phase_class(row["op"], row["subop"]),
            f"event {event_id} has invalid phase_class",
        )
        for field, expected_value in expected_context.items():
            require(
                row[field] == expected_value,
                f"event {event_id} has invalid {field}=<{row[field]}>, "
                f"expected <{expected_value}>",
            )
        require(
            row["offset_feature"] == offset_feature(row),
            f"event {event_id} has invalid offset_feature",
        )
        require(
            row["call_context"] == call_context(row),
            f"event {event_id} has invalid call_context",
        )
        require(
            row["physical_dpu_identity"] == physical_dpu_identity(row),
            f"event {event_id} has invalid physical_dpu_identity",
        )
        require(
            row["transport_key"] == transport_key(row),
            f"event {event_id} has invalid transport_key",
        )

    ops = Counter(row["op"] for row in rows)
    require(ops["dpu_alloc"] == 1, f"alloc={ops['dpu_alloc']}, expected 1")
    require(ops["dpu_load"] == 1, f"load={ops['dpu_load']}, expected 1")
    require(ops["dpu_copy_to"] == expected["copy_to"],
            f"copy_to={ops['dpu_copy_to']}, expected {expected['copy_to']}")
    require(ops["dpu_copy_from"] == expected["copy_from"],
            f"copy_from={ops['dpu_copy_from']}, expected {expected['copy_from']}")
    require(ops["dpu_launch"] == 1, f"launch={ops['dpu_launch']}, expected 1")
    require(ops["dpu_free"] == 1, f"free={ops['dpu_free']}, expected 1")
    require(rows[0]["op"] == "dpu_alloc", f"first op={rows[0]['op']}, expected dpu_alloc")
    require(rows[1]["op"] == "dpu_load", f"second op={rows[1]['op']}, expected dpu_load")
    require(rows[-1]["op"] == "dpu_free", f"last op={rows[-1]['op']}, expected dpu_free")
    wanted_sequence = expected_sequence(nr_dpus, expected["copy_from"])
    actual_sequence = [
        (row["op"], row["subop"], row["global_dpu_id"]) for row in rows
    ]
    for event_id, (actual, wanted) in enumerate(
        zip(actual_sequence, wanted_sequence)
    ):
        require(
            actual == wanted,
            f"event {event_id} semantic tuple={actual}, expected {wanted}",
        )

    subops = Counter(
        row["subop"]
        for row in rows
        if row["op"] in {"dpu_copy_to", "dpu_copy_from", "dpu_launch"}
    )
    require(subops == Counter(expected["subops"]),
            f"subop counts={dict(subops)}, expected {expected['subops']}")

    lifecycle_rows = [
        row for row in rows if row["op"] in {"dpu_alloc", "dpu_load", "dpu_free"}
    ]
    non_transfer_rows = [
        row
        for row in rows
        if row["op"] not in {"dpu_copy_to", "dpu_copy_from"}
    ]
    non_transfer_empty_fields = (
        "direction",
        "global_dpu_id",
        "rank_ordinal",
        "dpu_id_in_rank",
        "sdk_physical_rank_id",
        "dpu_sysfs_rank_id",
        "dpu_rank_numa_node",
        "dpu_channel_id",
        "sdk_slice_id",
        "sdk_member_id",
        "dpu_ci_id",
        "dpu_member_id",
        "physical_dpu_identity",
        "cpu_dpu_numa_relation",
        "sdk_api_kind",
        "logical_distribution_class",
        "target_space",
        "transfer_bytes_per_dpu",
        "active_dpus",
        "active_ranks",
        "active_dpus_per_rank",
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
            for row in non_transfer_rows
            for field in non_transfer_empty_fields
        ),
        "non-transfer event contains transfer metadata",
    )
    require(
        all(row["offset_feature"] == "none" for row in non_transfer_rows),
        "non-transfer event has invalid offset_feature",
    )
    require(
        all(int(row["measured_ns"]) > 0 for row in lifecycle_rows),
        "lifecycle event has a non-positive duration",
    )

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
    observed_dpu_numa_nodes = {
        int(row["dpu_rank_numa_node"]) for row in copy_rows
    }
    observed_sysfs_ranks = {
        int(row["dpu_sysfs_rank_id"]) for row in copy_rows
    }
    if expected_dpu_numa_node is not None:
        require(
            observed_dpu_numa_nodes == {expected_dpu_numa_node},
            f"DPU NUMA nodes={observed_dpu_numa_nodes}, expected "
            f"{expected_dpu_numa_node}",
        )
    if expected_sysfs_ranks is not None:
        require(
            observed_sysfs_ranks == expected_sysfs_ranks,
            f"sysfs ranks={sorted(observed_sysfs_ranks)}, expected "
            f"{sorted(expected_sysfs_ranks)}",
        )
    for row in copy_rows:
        event_id = row["event_id"]
        global_dpu_id = int(row["global_dpu_id"])
        require(
            row["target_space"] == "MRAM"
            and row["target_symbol"] == "DPU_MRAM_HEAP_POINTER_NAME",
            f"event {event_id} has invalid target",
        )
        require(
            row["direction"]
            == ("TO_DPU" if row["op"] == "dpu_copy_to" else "FROM_DPU"),
            f"event {event_id} has invalid direction",
        )
        require(
            int(row["transfer_bytes_per_dpu"])
            == int(row["transfer_bytes"]),
            f"event {event_id} has inconsistent transfer byte fields",
        )
        require(
            row["active_dpus"] == "1"
            and row["active_ranks"] == "1"
            and row["active_dpus_per_rank"] == "1",
            f"event {event_id} has invalid single-copy active topology",
        )
        require(
            int(row["rank_ordinal"]) == global_dpu_id // 64
            and int(row["dpu_id_in_rank"]) == global_dpu_id % 64,
            f"event {event_id} has inconsistent DPU topology",
        )
        require(
            row["dpu_ci_id"] == row["sdk_slice_id"]
            and row["dpu_member_id"] == row["sdk_member_id"],
            f"event {event_id} has inconsistent CI/member aliases",
        )
        require(
            row["cpu_dpu_numa_relation"]
            == expected_cpu_dpu_numa_relation(row),
            f"event {event_id} has invalid cpu_dpu_numa_relation",
        )
        require(
            int(row["logical_bytes"]) <= int(row["transfer_bytes"])
            and int(row["transfer_bytes"]) > 0
            and int(row["transfer_bytes"]) % 8 == 0,
            f"event {event_id} has invalid logical/aligned byte values",
        )
        require(
            int(row["offset_bytes"]) % 8 == 0,
            f"event {event_id} has unaligned MRAM offset",
        )

    return {
        "run_id": run_ids.pop(),
        "repeat_id": repeats.pop(),
        "configured_dpus": nr_dpus,
        "num_tasklets": tasklets.pop(),
        "actual_ranks": rank_count,
        "events": len(rows),
        "h2d_bytes": h2d,
        "d2h_bytes": d2h,
        "alloc_measured_ns": sum(
            int(row["measured_ns"]) for row in rows if row["op"] == "dpu_alloc"
        ),
        "load_measured_ns": sum(
            int(row["measured_ns"]) for row in rows if row["op"] == "dpu_load"
        ),
        "h2d_measured_ns": sum(
            int(row["measured_ns"]) for row in rows if row["op"] == "dpu_copy_to"
        ),
        "d2h_measured_ns": sum(
            int(row["measured_ns"]) for row in rows if row["op"] == "dpu_copy_from"
        ),
        "launch_measured_ns": sum(
            int(row["measured_ns"]) for row in rows if row["op"] == "dpu_launch"
        ),
        "free_measured_ns": sum(
            int(row["measured_ns"]) for row in rows if row["op"] == "dpu_free"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--expected-host-numa-node", type=int)
    parser.add_argument("--expected-dpu-numa-node", type=int)
    parser.add_argument("--expected-sysfs-ranks", type=parse_int_set)
    args = parser.parse_args()

    summaries = []
    failed = False
    for path in args.traces:
        try:
            summary = validate(
                path,
                expected_host_numa_node=args.expected_host_numa_node,
                expected_dpu_numa_node=args.expected_dpu_numa_node,
                expected_sysfs_ranks=args.expected_sysfs_ranks,
            )
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
        for key in (
            "alloc_measured_ns",
            "load_measured_ns",
            "h2d_measured_ns",
            "launch_measured_ns",
            "d2h_measured_ns",
            "free_measured_ns",
        ):
            values = [int(summary[key]) for summary in summaries]
            print(f"{key}_median={statistics.median(values):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
