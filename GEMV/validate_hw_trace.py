#!/usr/bin/env python3
"""Validate GEMV v9 event traces and their per-DPU detail files."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from transport_key import (
    logical_distribution_class,
    phase_class,
    previous_sdk_op_class,
    same_source_across_group,
    sdk_api_kind,
    source_buffer_reuse_class,
    target_region_reuse_class,
    transport_key,
)


SUPPORTED_DPUS = {64, 128, 256, 512, 1024, 1216}
ITERATIONS = 4
M_SIZE = 8192
N_SIZE = 8192
ELEMENT_BYTES = 4

EVENT_FIELDS = {
    "run_id",
    "repeat_id",
    "event_id",
    "configured_dpus",
    "actual_ranks",
    "num_tasklets",
    "m_size",
    "n_size",
    "n_size_pad",
    "max_rows_per_dpu",
    "op",
    "direction",
    "sdk_api_kind",
    "timing_scope",
    "logical_distribution_class",
    "target_space",
    "transfer_bytes_per_dpu",
    "active_dpus",
    "active_ranks",
    "active_dpus_per_rank",
    "rank_ordinal",
    "dpu_id_in_rank",
    "same_source_across_group",
    "host_numa_node",
    "dpu_rank_numa_nodes",
    "cpu_dpu_numa_relation",
    "dpu_channel_ids",
    "dpu_sysfs_rank_ids",
    "sdk_physical_rank_ids",
    "dpu_ci_ids",
    "dpu_member_ids",
    "allocated_topology_signature",
    "allocated_dpus",
    "allocated_ranks",
    "previous_sdk_op",
    "previous_sdk_direction",
    "previous_sdk_transfer_bytes",
    "previous_sdk_mux_domain_class",
    "previous_sdk_subop",
    "previous_sdk_target_space",
    "previous_sdk_op_class",
    "source_buffer_reuse_class",
    "target_region_reuse_class",
    "source_buffer_use_count_before",
    "target_region_access_count_before",
    "phase_class",
    "subop",
    "iteration",
    "warmup",
    "size_per_dpu_bytes",
    "total_logical_bytes",
    "total_transfer_bytes",
    "target_symbol",
    "offset_bytes",
    "process_state",
    "host_binding_mode",
    "host_cpu_list",
    "pretrace_warmup_runs",
    "transport_key",
    "host_start_ns",
    "host_end_ns",
    "measured_ns",
    "thread_cpu_ns",
    "wall_minus_thread_cpu_ns",
    "cpu_id_start",
    "cpu_id_end",
    "voluntary_context_switch_delta",
    "involuntary_context_switch_delta",
    "minor_fault_delta",
    "major_fault_delta",
}

DPU_FIELDS = {
    "run_id",
    "repeat_id",
    "event_id",
    "configured_dpus",
    "actual_ranks",
    "num_tasklets",
    "iteration",
    "warmup",
    "op",
    "subop",
    "direction",
    "global_dpu_id",
    "rank_ordinal",
    "dpu_id_in_rank",
    "sdk_physical_rank_id",
    "dpu_sysfs_rank_id",
    "dpu_rank_numa_node",
    "dpu_channel_id",
    "dpu_ci_id",
    "dpu_member_id",
    "logical_bytes",
    "transfer_bytes",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_int_set(value: str) -> set[int]:
    return {int(token) for token in value.split(",") if token.strip()}


def read_csv(
    path: Path, expected_fields: set[str], kind: str
) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        require(
            fields == expected_fields,
            f"{kind} header fields differ: got {sorted(fields)}",
        )
        rows = list(reader)
    require(bool(rows), f"{kind} trace contains zero rows")
    return rows


def one_int(rows: list[dict[str, str]], field: str) -> int:
    values = {int(row[field]) for row in rows}
    require(len(values) == 1, f"multiple {field} values: {sorted(values)}")
    return values.pop()


def one_text(rows: list[dict[str, str]], field: str) -> str:
    values = {row[field] for row in rows}
    require(len(values) == 1, f"multiple {field} values: {sorted(values)}")
    return values.pop()


def detail_path_for(event_path: Path) -> Path:
    require(event_path.suffix == ".csv", f"trace path needs .csv: {event_path}")
    return event_path.with_name(f"{event_path.stem}_dpus.csv")


def expected_rows_per_dpu(nr_dpus: int) -> list[int]:
    chunks, extra = divmod(M_SIZE, nr_dpus)
    return [chunks + int(dpu_id < extra) for dpu_id in range(nr_dpus)]


def expected_sequence() -> list[tuple[str, str, str, str, str]]:
    sequence = [
        ("dpu_alloc", "", "", "", ""),
        ("dpu_load", "", "", "", ""),
    ]
    for iteration in range(ITERATIONS):
        warmup = "1" if iteration == 0 else "0"
        for subop, direction in (
            ("input_arguments", "TO_DPU"),
            ("input_matrix", "TO_DPU"),
            ("input_vector", "TO_DPU"),
        ):
            sequence.append(
                ("dpu_push_xfer", subop, direction, str(iteration), warmup)
            )
        sequence.append(("dpu_launch", "sync", "", str(iteration), warmup))
        sequence.append(
            (
                "dpu_push_xfer",
                "output_vector",
                "FROM_DPU",
                str(iteration),
                warmup,
            )
        )
    sequence.append(("dpu_free", "", "", "", ""))
    return sequence


def expected_transfer(
    subop: str, nr_dpus: int, max_rows: int, n_size_pad: int
) -> dict[str, object]:
    rows = expected_rows_per_dpu(nr_dpus)
    if subop == "input_arguments":
        logical = [16] * nr_dpus
        size = 16
        target_space = "WRAM"
        target_symbol = "DPU_INPUT_ARGUMENTS"
        offset = 0
    elif subop == "input_matrix":
        logical = [row_count * N_SIZE * ELEMENT_BYTES for row_count in rows]
        size = max_rows * n_size_pad * ELEMENT_BYTES
        target_space = "MRAM"
        target_symbol = "DPU_MRAM_HEAP_POINTER_NAME"
        offset = 0
    elif subop == "input_vector":
        logical = [N_SIZE * ELEMENT_BYTES] * nr_dpus
        size = n_size_pad * ELEMENT_BYTES
        target_space = "MRAM"
        target_symbol = "DPU_MRAM_HEAP_POINTER_NAME"
        offset = max_rows * n_size_pad * ELEMENT_BYTES
    elif subop == "output_vector":
        logical = [row_count * ELEMENT_BYTES for row_count in rows]
        size = max_rows * ELEMENT_BYTES
        target_space = "MRAM"
        target_symbol = "DPU_MRAM_HEAP_POINTER_NAME"
        offset = (
            max_rows * n_size_pad * ELEMENT_BYTES + n_size_pad * ELEMENT_BYTES
        )
    else:
        raise ValueError(f"unknown GEMV subop={subop}")
    return {
        "logical": logical,
        "size": size,
        "target_space": target_space,
        "target_symbol": target_symbol,
        "offset": offset,
    }


def topology_from_details(
    details: list[dict[str, str]], nr_dpus: int, ranks: int
) -> dict[str, str]:
    first_event = min(int(row["event_id"]) for row in details)
    rows = [row for row in details if int(row["event_id"]) == first_event]
    require(len(rows) == nr_dpus, "first transfer has an incomplete DPU detail set")
    rows.sort(key=lambda row: int(row["global_dpu_id"]))
    require(
        [int(row["global_dpu_id"]) for row in rows] == list(range(nr_dpus)),
        "global DPU IDs differ from a contiguous sequence",
    )
    rank_rows: list[dict[str, str]] = []
    for rank_ordinal in range(ranks):
        members = [
            row for row in rows if int(row["rank_ordinal"]) == rank_ordinal
        ]
        require(len(members) == 64, f"rank {rank_ordinal} has {len(members)} DPUs")
        require(
            {int(row["dpu_id_in_rank"]) for row in members} == set(range(64)),
            f"rank {rank_ordinal} DPU IDs differ from 0..63",
        )
        for field in (
            "sdk_physical_rank_id",
            "dpu_sysfs_rank_id",
            "dpu_rank_numa_node",
            "dpu_channel_id",
        ):
            require(
                len({row[field] for row in members}) == 1,
                f"rank {rank_ordinal} has multiple {field} values",
            )
        require(
            {int(row["dpu_ci_id"]) for row in members} == set(range(8)),
            f"rank {rank_ordinal} CI IDs differ from 0..7",
        )
        require(
            {int(row["dpu_member_id"]) for row in members} == set(range(8)),
            f"rank {rank_ordinal} member IDs differ from 0..7",
        )
        rank_rows.append(members[0])

    def joined(field: str) -> str:
        return "|".join(row[field] for row in rank_rows)

    signature = "|".join(
        f"r{row['dpu_sysfs_rank_id']}@n{row['dpu_rank_numa_node']}@c{row['dpu_channel_id']}"
        for row in rank_rows
    )
    return {
        "active_dpus_per_rank": "|".join("64" for _ in rank_rows),
        "dpu_rank_numa_nodes": joined("dpu_rank_numa_node"),
        "dpu_channel_ids": joined("dpu_channel_id"),
        "dpu_sysfs_rank_ids": joined("dpu_sysfs_rank_id"),
        "sdk_physical_rank_ids": joined("sdk_physical_rank_id"),
        "allocated_topology_signature": signature,
    }


def validate(
    event_path: Path,
    expected_host_numa_node: int = 0,
    expected_dpu_numa_node: int = 0,
    expected_sysfs_ranks: set[int] | None = None,
    expected_host_binding_mode: str | None = None,
    expected_host_cpu_list: str | None = None,
) -> dict[str, object]:
    details_path = detail_path_for(event_path)
    events = read_csv(event_path, EVENT_FIELDS, "event")
    details = read_csv(details_path, DPU_FIELDS, "DPU")
    nr_dpus = one_int(events, "configured_dpus")
    ranks = one_int(events, "actual_ranks")
    tasklets = one_int(events, "num_tasklets")
    repeat_id = one_int(events, "repeat_id")
    run_id = one_text(events, "run_id")

    require(nr_dpus in SUPPORTED_DPUS, f"unsupported configured_dpus={nr_dpus}")
    require(ranks == nr_dpus // 64, f"actual_ranks={ranks}, expected {nr_dpus // 64}")
    require(one_int(events, "m_size") == M_SIZE, f"m_size differs from {M_SIZE}")
    require(one_int(events, "n_size") == N_SIZE, f"n_size differs from {N_SIZE}")
    n_size_pad = one_int(events, "n_size_pad")
    require(n_size_pad == N_SIZE, f"n_size_pad={n_size_pad}, expected {N_SIZE}")
    max_rows = max(expected_rows_per_dpu(nr_dpus))
    if max_rows % 2 == 1:
        max_rows += 1
    require(
        one_int(events, "max_rows_per_dpu") == max_rows,
        f"max_rows_per_dpu differs from {max_rows}",
    )
    require(tasklets > 0, "num_tasklets must be positive")
    require(
        one_text(events, "host_numa_node") == str(expected_host_numa_node),
        "host NUMA node differs from the requested node",
    )
    require(bool(one_text(events, "process_state")), "process_state is empty")
    host_binding_mode = one_text(events, "host_binding_mode")
    host_cpu_list = one_text(events, "host_cpu_list")
    require(bool(host_binding_mode), "host_binding_mode is empty")
    require(bool(host_cpu_list), "host_cpu_list is empty")
    if expected_host_binding_mode is not None:
        require(
            host_binding_mode == expected_host_binding_mode,
            "host binding mode differs from the requested mode",
        )
    if expected_host_cpu_list is not None:
        require(
            host_cpu_list == expected_host_cpu_list,
            "host CPU list differs from the requested list",
        )

    wanted_sequence = expected_sequence()
    require(len(events) == len(wanted_sequence), f"event rows={len(events)}, expected 23")
    require(
        [int(row["event_id"]) for row in events] == list(range(len(events))),
        "event IDs differ from contiguous zero-based IDs",
    )
    actual_sequence = [
        (
            row["op"],
            row["subop"],
            row["direction"],
            row["iteration"],
            row["warmup"],
        )
        for row in events
    ]
    require(actual_sequence == wanted_sequence, "event semantic sequence differs")

    require(len(details) == ITERATIONS * 4 * nr_dpus, "DPU detail row count differs")
    for field, expected in (
        ("configured_dpus", nr_dpus),
        ("actual_ranks", ranks),
        ("num_tasklets", tasklets),
        ("repeat_id", repeat_id),
    ):
        require(one_int(details, field) == expected, f"DPU detail {field} differs")
    require(one_text(details, "run_id") == run_id, "DPU detail run_id differs")

    topology = topology_from_details(details, nr_dpus, ranks)
    observed_sysfs = {int(token) for token in topology["dpu_sysfs_rank_ids"].split("|")}
    observed_numas = {int(token) for token in topology["dpu_rank_numa_nodes"].split("|")}
    require(
        observed_numas == {expected_dpu_numa_node},
        f"DPU NUMA nodes={sorted(observed_numas)}, expected {expected_dpu_numa_node}",
    )
    if expected_sysfs_ranks is not None:
        require(
            observed_sysfs == expected_sysfs_ranks,
            f"sysfs ranks={sorted(observed_sysfs)}, expected {sorted(expected_sysfs_ranks)}",
        )

    details_by_event: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in details:
        details_by_event[int(row["event_id"])].append(row)

    transfer_rows = [row for row in events if row["op"] == "dpu_push_xfer"]
    nontransfer_rows = [row for row in events if row["op"] != "dpu_push_xfer"]
    for row in events:
        event_id = int(row["event_id"])
        start = int(row["host_start_ns"])
        end = int(row["host_end_ns"])
        measured = int(row["measured_ns"])
        thread_cpu = int(row["thread_cpu_ns"])
        wait_like = int(row["wall_minus_thread_cpu_ns"])
        cpu_start = int(row["cpu_id_start"])
        cpu_end = int(row["cpu_id_end"])
        require(end >= start, f"event {event_id} has a negative interval")
        require(measured == end - start, f"event {event_id} measured_ns differs")
        require(measured > 0, f"event {event_id} measured_ns is zero")
        require(thread_cpu >= 0, f"event {event_id} thread_cpu_ns is negative")
        require(
            wait_like == max(0, measured - thread_cpu),
            f"event {event_id} wall-minus-thread CPU time differs",
        )
        require(cpu_start >= 0 and cpu_end >= 0, f"event {event_id} CPU ID differs")
        for field in (
            "voluntary_context_switch_delta",
            "involuntary_context_switch_delta",
            "minor_fault_delta",
            "major_fault_delta",
        ):
            require(int(row[field]) >= 0, f"event {event_id} {field} is negative")
        require(
            row["sdk_api_kind"] == sdk_api_kind(row["op"]),
            f"event {event_id} sdk_api_kind differs",
        )
        require(
            row["logical_distribution_class"]
            == logical_distribution_class(row["op"], row["subop"]),
            f"event {event_id} distribution class differs",
        )
        require(
            row["same_source_across_group"]
            == same_source_across_group(row["op"], row["subop"]),
            f"event {event_id} source reuse class differs",
        )
        require(
            row["phase_class"] == phase_class(row["op"], row["warmup"]),
            f"event {event_id} phase class differs",
        )
        require(
            row["transport_key"] == transport_key(row),
            f"event {event_id} v9 transport key differs from raw fields",
        )

    transfer_only_fields = (
        "direction",
        "sdk_api_kind",
        "timing_scope",
        "logical_distribution_class",
        "target_space",
        "transfer_bytes_per_dpu",
        "active_ranks",
        "active_dpus_per_rank",
        "rank_ordinal",
        "dpu_id_in_rank",
        "same_source_across_group",
        "dpu_rank_numa_nodes",
        "cpu_dpu_numa_relation",
        "dpu_channel_ids",
        "dpu_sysfs_rank_ids",
        "sdk_physical_rank_ids",
        "dpu_ci_ids",
        "dpu_member_ids",
        "allocated_topology_signature",
        "allocated_dpus",
        "allocated_ranks",
        "previous_sdk_op",
        "previous_sdk_direction",
        "previous_sdk_transfer_bytes",
        "previous_sdk_mux_domain_class",
        "previous_sdk_subop",
        "previous_sdk_target_space",
        "previous_sdk_op_class",
        "source_buffer_reuse_class",
        "target_region_reuse_class",
        "source_buffer_use_count_before",
        "target_region_access_count_before",
        "phase_class",
        "size_per_dpu_bytes",
        "total_logical_bytes",
        "total_transfer_bytes",
        "target_symbol",
        "offset_bytes",
        "transport_key",
    )
    require(
        all(row[field] == "" for row in nontransfer_rows for field in transfer_only_fields),
        "lifecycle or launch event carries transfer metadata",
    )

    durations_by_subop: dict[str, list[int]] = defaultdict(list)
    for row in transfer_rows:
        event_id = int(row["event_id"])
        iteration = int(row["iteration"])
        previous = events[event_id - 1] if event_id > 0 else None
        expectation = expected_transfer(row["subop"], nr_dpus, max_rows, n_size_pad)
        size = int(expectation["size"])
        logical = list(expectation["logical"])
        require(row["sdk_api_kind"] == "PUSH_XFER", f"event {event_id} API kind differs")
        require(row["timing_scope"] == "PUSH_ONLY", f"event {event_id} timing scope differs")
        require(int(row["transfer_bytes_per_dpu"]) == size, f"event {event_id} size differs")
        require(int(row["active_dpus"]) == nr_dpus, f"event {event_id} active DPUs differ")
        require(int(row["active_ranks"]) == ranks, f"event {event_id} active ranks differ")
        require(row["rank_ordinal"] == "ALL", f"event {event_id} rank marker differs")
        require(row["dpu_id_in_rank"] == "ALL", f"event {event_id} DPU marker differs")
        require(row["cpu_dpu_numa_relation"] == "LOCAL", f"event {event_id} NUMA relation differs")
        require(row["dpu_ci_ids"] == "0-7", f"event {event_id} CI domain differs")
        require(row["dpu_member_ids"] == "0-7", f"event {event_id} member domain differs")
        require(int(row["allocated_dpus"]) == nr_dpus, f"event {event_id} allocated DPUs differ")
        require(int(row["allocated_ranks"]) == ranks, f"event {event_id} allocated ranks differ")
        require(
            row["previous_sdk_mux_domain_class"] == "COLLECTION",
            f"event {event_id} predecessor MUX class differs",
        )
        require(
            row["previous_sdk_op_class"] == previous_sdk_op_class(row),
            f"event {event_id} predecessor operation class differs",
        )
        require(
            row["previous_sdk_subop"]
            == ("NONE" if previous is None or not previous["subop"] else previous["subop"]),
            f"event {event_id} predecessor subop differs",
        )
        require(
            row["previous_sdk_target_space"]
            == (
                "NONE"
                if previous is None or not previous["target_space"]
                else previous["target_space"]
            ),
            f"event {event_id} predecessor target space differs",
        )
        require(
            int(row["source_buffer_use_count_before"]) == iteration,
            f"event {event_id} source buffer use count differs",
        )
        require(
            int(row["target_region_access_count_before"]) == iteration,
            f"event {event_id} target region access count differs",
        )
        require(
            row["source_buffer_reuse_class"]
            == source_buffer_reuse_class(iteration),
            f"event {event_id} source buffer reuse class differs",
        )
        require(
            row["target_region_reuse_class"]
            == target_region_reuse_class(iteration),
            f"event {event_id} target region reuse class differs",
        )
        for field, expected in topology.items():
            require(row[field] == expected, f"event {event_id} {field} differs")
        require(row["target_space"] == expectation["target_space"], f"event {event_id} target space differs")
        require(row["target_symbol"] == expectation["target_symbol"], f"event {event_id} target symbol differs")
        require(int(row["offset_bytes"]) == expectation["offset"], f"event {event_id} offset differs")
        require(int(row["size_per_dpu_bytes"]) == size, f"event {event_id} per-DPU size differs")
        require(int(row["total_logical_bytes"]) == sum(logical), f"event {event_id} logical total differs")
        require(int(row["total_transfer_bytes"]) == size * nr_dpus, f"event {event_id} physical total differs")

        event_details = details_by_event[event_id]
        require(len(event_details) == nr_dpus, f"event {event_id} DPU rows differ")
        event_details.sort(key=lambda item: int(item["global_dpu_id"]))
        require(
            [int(item["logical_bytes"]) for item in event_details] == logical,
            f"event {event_id} per-DPU logical distribution differs",
        )
        require(
            {int(item["transfer_bytes"]) for item in event_details} == {size},
            f"event {event_id} per-DPU physical bytes differ",
        )
        durations_by_subop[row["subop"]].append(int(row["measured_ns"]))

    return {
        "path": str(event_path),
        "configured_dpus": nr_dpus,
        "actual_ranks": ranks,
        "num_tasklets": tasklets,
        "event_rows": len(events),
        "transfer_rows": len(transfer_rows),
        "dpu_detail_rows": len(details),
        "sysfs_ranks": topology["dpu_sysfs_rank_ids"],
        "channels": topology["dpu_channel_ids"],
        "transfer_median_ns": round(
            statistics.median(
                duration for values in durations_by_subop.values() for duration in values
            )
        ),
        "host_binding_mode": host_binding_mode,
        "host_cpu_list": host_cpu_list,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--expected-host-numa-node", type=int, default=0)
    parser.add_argument("--expected-dpu-numa-node", type=int, default=0)
    parser.add_argument("--expected-sysfs-ranks", type=parse_int_set)
    parser.add_argument("--expected-host-binding-mode")
    parser.add_argument("--expected-host-cpu-list")
    args = parser.parse_args()
    try:
        summaries = [
            validate(
                path,
                args.expected_host_numa_node,
                args.expected_dpu_numa_node,
                args.expected_sysfs_ranks,
                args.expected_host_binding_mode,
                args.expected_host_cpu_list,
            )
            for path in args.traces
        ]
    except (OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    for summary in summaries:
        print(
            "PASS "
            f"{summary['path']} dpus={summary['configured_dpus']} "
            f"ranks={summary['actual_ranks']} events={summary['event_rows']} "
            f"transfer_rows={summary['transfer_rows']} "
            f"detail_rows={summary['dpu_detail_rows']} "
            f"median_ns={summary['transfer_median_ns']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
