#!/usr/bin/env python3
"""Validate SCAN-SSA v9 collection-transfer traces."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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


SUPPORTED_DPUS = {64, 128, 256, 512, 1024, 1152}
ITERATIONS = 4
INPUT_SIZE = 251658240
TASKLETS = 16
BLOCK_LOG2 = 10
ELEMENT_BYTES = 8
REGS = (1 << BLOCK_LOG2) // ELEMENT_BYTES
ROUND_ELEMENTS = TASKLETS * REGS
ARGUMENT_BYTES = 16
RESULT_BYTES_PER_TASKLET = 8

EVENT_FIELDS = {
    "run_id", "repeat_id", "event_id", "configured_dpus", "actual_ranks",
    "num_tasklets", "input_size", "input_size_per_dpu",
    "input_size_per_dpu_round", "scaling_mode", "op", "direction",
    "sdk_api_kind", "timing_scope", "logical_distribution_class",
    "target_space", "transfer_bytes_per_dpu", "active_dpus", "active_ranks",
    "active_dpus_per_rank", "rank_ordinal", "dpu_id_in_rank",
    "same_source_across_group", "host_numa_node", "dpu_rank_numa_nodes",
    "cpu_dpu_numa_relation", "dpu_channel_ids", "dpu_sysfs_rank_ids",
    "sdk_physical_rank_ids", "dpu_ci_ids", "dpu_member_ids",
    "allocated_topology_signature", "allocated_dpus", "allocated_ranks",
    "previous_sdk_op", "previous_sdk_direction",
    "previous_sdk_transfer_bytes", "previous_sdk_mux_domain_class",
    "previous_sdk_subop", "previous_sdk_target_space",
    "previous_sdk_op_class", "source_buffer_reuse_class",
    "target_region_reuse_class", "source_buffer_use_count_before",
    "target_region_access_count_before", "phase_class", "subop",
    "iteration", "warmup", "size_per_dpu_bytes", "total_logical_bytes",
    "total_transfer_bytes", "target_symbol", "offset_bytes", "process_state",
    "host_binding_mode", "host_cpu_list", "algorithm_variant",
    "scaling_mode_label", "diagnostic_copy_ordinal",
    "mram_push_ordinal_since_launch", "replay_delay_requested_us",
    "pretrace_warmup_runs", "transport_key", "host_start_ns", "host_end_ns",
    "measured_ns", "thread_cpu_ns", "wall_minus_thread_cpu_ns",
    "cpu_id_start", "cpu_id_end", "voluntary_context_switch_delta",
    "involuntary_context_switch_delta", "minor_fault_delta",
    "major_fault_delta",
}

DPU_FIELDS = {
    "run_id", "repeat_id", "event_id", "configured_dpus", "actual_ranks",
    "num_tasklets", "iteration", "warmup", "op", "subop", "direction",
    "global_dpu_id", "rank_ordinal", "dpu_id_in_rank",
    "sdk_physical_rank_id", "dpu_sysfs_rank_id", "dpu_rank_numa_node",
    "dpu_channel_id", "dpu_ci_id", "dpu_member_id", "logical_bytes",
    "transfer_bytes",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_int_set(value: str) -> Set[int]:
    return {int(token) for token in value.split(",") if token.strip()}


def read_csv(
    path: Path, expected_fields: Set[str], kind: str
) -> List[Dict[str, str]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        require(
            fields == expected_fields,
            "{} header fields differ: got {}".format(kind, sorted(fields)),
        )
        rows = list(reader)
    require(bool(rows), "{} trace contains zero rows".format(kind))
    return rows


def one_int(rows: List[Dict[str, str]], field: str) -> int:
    values = {int(row[field]) for row in rows}
    require(len(values) == 1, "multiple {} values: {}".format(field, sorted(values)))
    return values.pop()


def one_text(rows: List[Dict[str, str]], field: str) -> str:
    values = {row[field] for row in rows}
    require(len(values) == 1, "multiple {} values: {}".format(field, sorted(values)))
    return values.pop()


def detail_path_for(event_path: Path) -> Path:
    require(event_path.suffix == ".csv", "trace path needs .csv: {}".format(event_path))
    return event_path.with_name("{}_dpus.csv".format(event_path.stem))


def expected_layout(nr_dpus: int) -> Tuple[int, int, List[int]]:
    raw = (INPUT_SIZE + nr_dpus - 1) // nr_dpus
    rounded = (
        raw if raw % ROUND_ELEMENTS == 0
        else ((raw // ROUND_ELEMENTS) + 1) * ROUND_ELEMENTS
    )
    logical = []
    for dpu_id in range(nr_dpus):
        first = dpu_id * rounded
        remaining = max(0, INPUT_SIZE - first)
        logical.append(min(rounded, remaining) * ELEMENT_BYTES)
    return raw, rounded, logical


def expected_sequence() -> List[Tuple[str, str, str, str, str]]:
    sequence = [
        ("dpu_alloc", "", "", "", ""),
        ("dpu_load", "", "", "", ""),
    ]
    per_iteration = (
        ("dpu_push_xfer", "input_arguments_scan", "TO_DPU"),
        ("dpu_push_xfer", "input_data", "TO_DPU"),
        ("dpu_launch", "scan_sync", ""),
        ("dpu_push_xfer", "partial_results", "FROM_DPU"),
        ("dpu_push_xfer", "input_arguments_add", "TO_DPU"),
        ("dpu_launch", "add_sync", ""),
        ("dpu_push_xfer", "output_data", "FROM_DPU"),
    )
    for iteration in range(ITERATIONS):
        warmup = "1" if iteration == 0 else "0"
        sequence.extend(
            (op, subop, direction, str(iteration), warmup)
            for op, subop, direction in per_iteration
        )
    sequence.append(("dpu_free", "", "", "", ""))
    return sequence


def expected_transfer(
    subop: str, nr_dpus: int, rounded: int, data_logical: List[int]
) -> Dict[str, object]:
    if subop in {"input_arguments_scan", "input_arguments_add"}:
        return {
            "logical": [ARGUMENT_BYTES] * nr_dpus,
            "size": ARGUMENT_BYTES,
            "target_space": "WRAM",
            "target_symbol": "DPU_INPUT_ARGUMENTS",
            "offset": 0,
        }
    if subop == "partial_results":
        size = TASKLETS * RESULT_BYTES_PER_TASKLET
        return {
            "logical": [size] * nr_dpus,
            "size": size,
            "target_space": "WRAM",
            "target_symbol": "DPU_RESULTS",
            "offset": 0,
        }
    if subop in {"input_data", "output_data"}:
        size = rounded * ELEMENT_BYTES
        return {
            "logical": data_logical,
            "size": size,
            "target_space": "MRAM",
            "target_symbol": "DPU_MRAM_HEAP_POINTER_NAME",
            "offset": 0 if subop == "input_data" else size,
        }
    raise ValueError("unknown SCAN subop={}".format(subop))


def expected_reuse_counts(subop: str, iteration: int) -> Tuple[int, int]:
    source_count = iteration
    if subop == "input_arguments_scan":
        target_count = 2 * iteration
    elif subop == "input_arguments_add":
        target_count = 2 * iteration + 1
    else:
        target_count = iteration
    return source_count, target_count


def topology_from_details(
    details: List[Dict[str, str]], nr_dpus: int, ranks: int
) -> Dict[str, str]:
    first_event = min(int(row["event_id"]) for row in details)
    rows = [row for row in details if int(row["event_id"]) == first_event]
    require(len(rows) == nr_dpus, "first transfer has an incomplete DPU detail set")
    rows.sort(key=lambda row: int(row["global_dpu_id"]))
    require(
        [int(row["global_dpu_id"]) for row in rows] == list(range(nr_dpus)),
        "global DPU IDs differ from a contiguous sequence",
    )
    rank_rows = []
    for rank_ordinal in range(ranks):
        members = [
            row for row in rows if int(row["rank_ordinal"]) == rank_ordinal
        ]
        require(
            len(members) == 64,
            "rank {} has {} DPUs".format(rank_ordinal, len(members)),
        )
        require(
            {int(row["dpu_id_in_rank"]) for row in members} == set(range(64)),
            "rank {} DPU IDs differ from 0..63".format(rank_ordinal),
        )
        for field in (
            "sdk_physical_rank_id", "dpu_sysfs_rank_id",
            "dpu_rank_numa_node", "dpu_channel_id",
        ):
            require(
                len({row[field] for row in members}) == 1,
                "rank {} has multiple {} values".format(rank_ordinal, field),
            )
        require(
            {int(row["dpu_ci_id"]) for row in members} == set(range(8)),
            "rank {} CI IDs differ from 0..7".format(rank_ordinal),
        )
        require(
            {int(row["dpu_member_id"]) for row in members} == set(range(8)),
            "rank {} member IDs differ from 0..7".format(rank_ordinal),
        )
        rank_rows.append(members[0])

    def joined(field: str) -> str:
        return "|".join(row[field] for row in rank_rows)

    return {
        "active_dpus_per_rank": "|".join("64" for _ in rank_rows),
        "dpu_rank_numa_nodes": joined("dpu_rank_numa_node"),
        "dpu_channel_ids": joined("dpu_channel_id"),
        "dpu_sysfs_rank_ids": joined("dpu_sysfs_rank_id"),
        "sdk_physical_rank_ids": joined("sdk_physical_rank_id"),
        "allocated_topology_signature": "|".join(
            "r{}@n{}@c{}".format(
                row["dpu_sysfs_rank_id"], row["dpu_rank_numa_node"],
                row["dpu_channel_id"],
            )
            for row in rank_rows
        ),
    }


def validate(
    event_path: Path,
    expected_host_numa_node: int = 0,
    expected_dpu_numa_node: int = 0,
    expected_sysfs_ranks: Optional[Set[int]] = None,
    expected_host_binding_mode: Optional[str] = None,
    expected_host_cpu_list: Optional[str] = None,
) -> Dict[str, object]:
    events = read_csv(event_path, EVENT_FIELDS, "event")
    details = read_csv(detail_path_for(event_path), DPU_FIELDS, "DPU")
    nr_dpus = one_int(events, "configured_dpus")
    ranks = one_int(events, "actual_ranks")
    tasklets = one_int(events, "num_tasklets")
    repeat_id = one_int(events, "repeat_id")
    run_id = one_text(events, "run_id")
    raw, rounded, data_logical = expected_layout(nr_dpus)

    require(nr_dpus in SUPPORTED_DPUS, "unsupported configured_dpus={}".format(nr_dpus))
    require(ranks == nr_dpus // 64, "actual rank count differs")
    require(tasklets == TASKLETS, "num_tasklets differs from {}".format(TASKLETS))
    require(one_int(events, "input_size") == INPUT_SIZE, "input_size differs")
    require(
        one_int(events, "input_size_per_dpu") == raw,
        "input_size_per_dpu differs",
    )
    require(
        one_int(events, "input_size_per_dpu_round") == rounded,
        "input_size_per_dpu_round differs",
    )
    require(one_text(events, "scaling_mode") == "STRONG", "scaling mode differs")
    require(
        one_text(events, "scaling_mode_label") == "STRONG",
        "scaling mode label differs",
    )
    require(
        one_text(events, "algorithm_variant") == "SCAN_SSA",
        "algorithm variant differs",
    )
    require(
        one_text(events, "host_numa_node") == str(expected_host_numa_node),
        "host NUMA node differs",
    )
    require(bool(one_text(events, "process_state")), "process_state is empty")
    host_binding_mode = one_text(events, "host_binding_mode")
    host_cpu_list = one_text(events, "host_cpu_list")
    require(bool(host_binding_mode), "host_binding_mode is empty")
    require(bool(host_cpu_list), "host_cpu_list is empty")
    if expected_host_binding_mode is not None:
        require(
            host_binding_mode == expected_host_binding_mode,
            "host binding mode differs",
        )
    if expected_host_cpu_list is not None:
        require(host_cpu_list == expected_host_cpu_list, "host CPU list differs")

    wanted_sequence = expected_sequence()
    require(
        len(events) == len(wanted_sequence),
        "event rows={}, expected {}".format(len(events), len(wanted_sequence)),
    )
    require(
        [int(row["event_id"]) for row in events] == list(range(len(events))),
        "event IDs differ from contiguous zero-based IDs",
    )
    actual_sequence = [
        (row["op"], row["subop"], row["direction"], row["iteration"], row["warmup"])
        for row in events
    ]
    require(actual_sequence == wanted_sequence, "event semantic sequence differs")

    transfer_count = sum(item[0] == "dpu_push_xfer" for item in wanted_sequence)
    require(
        len(details) == transfer_count * nr_dpus,
        "DPU detail row count differs",
    )
    for field, expected in (
        ("configured_dpus", nr_dpus), ("actual_ranks", ranks),
        ("num_tasklets", tasklets), ("repeat_id", repeat_id),
    ):
        require(one_int(details, field) == expected, "DPU detail {} differs".format(field))
    require(one_text(details, "run_id") == run_id, "DPU detail run_id differs")

    topology = topology_from_details(details, nr_dpus, ranks)
    observed_sysfs = {
        int(token) for token in topology["dpu_sysfs_rank_ids"].split("|")
    }
    observed_numas = {
        int(token) for token in topology["dpu_rank_numa_nodes"].split("|")
    }
    require(
        observed_numas == {expected_dpu_numa_node},
        "DPU NUMA nodes differ",
    )
    if expected_sysfs_ranks is not None:
        require(observed_sysfs == expected_sysfs_ranks, "sysfs ranks differ")
    require(
        observed_sysfs.isdisjoint({4, 5}),
        "faulty sysfs rank 4 or 5 is present",
    )

    details_by_event = defaultdict(list)
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
        require(end >= start, "event {} has a negative interval".format(event_id))
        require(measured == end - start, "event {} measured_ns differs".format(event_id))
        require(measured > 0, "event {} measured_ns is zero".format(event_id))
        require(
            int(row["wall_minus_thread_cpu_ns"]) == max(0, measured - thread_cpu),
            "event {} wall-minus-thread CPU time differs".format(event_id),
        )
        require(int(row["cpu_id_start"]) >= 0, "event CPU start differs")
        require(int(row["cpu_id_end"]) >= 0, "event CPU end differs")
        require(
            row["sdk_api_kind"] == sdk_api_kind(row["op"]),
            "event {} sdk_api_kind differs".format(event_id),
        )
        require(
            row["logical_distribution_class"]
            == logical_distribution_class(row["op"], row["subop"]),
            "event {} distribution class differs".format(event_id),
        )
        require(
            row["same_source_across_group"]
            == same_source_across_group(row["op"], row["subop"]),
            "event {} same-source field differs".format(event_id),
        )
        require(
            row["phase_class"] == phase_class(row["op"], row["warmup"]),
            "event {} phase class differs".format(event_id),
        )
        require(
            row["transport_key"] == transport_key(row),
            "event {} transport key differs".format(event_id),
        )

    transfer_only_fields = (
        "direction", "sdk_api_kind", "timing_scope",
        "logical_distribution_class", "target_space",
        "transfer_bytes_per_dpu", "active_dpus", "active_ranks",
        "active_dpus_per_rank", "rank_ordinal", "dpu_id_in_rank",
        "same_source_across_group", "dpu_rank_numa_nodes",
        "cpu_dpu_numa_relation", "dpu_channel_ids", "dpu_sysfs_rank_ids",
        "sdk_physical_rank_ids", "dpu_ci_ids", "dpu_member_ids",
        "allocated_topology_signature", "allocated_dpus", "allocated_ranks",
        "previous_sdk_op", "previous_sdk_direction",
        "previous_sdk_transfer_bytes", "previous_sdk_mux_domain_class",
        "previous_sdk_subop", "previous_sdk_target_space",
        "previous_sdk_op_class", "source_buffer_reuse_class",
        "target_region_reuse_class", "source_buffer_use_count_before",
        "target_region_access_count_before", "phase_class",
        "size_per_dpu_bytes", "total_logical_bytes",
        "total_transfer_bytes", "target_symbol", "offset_bytes",
        "transport_key", "diagnostic_copy_ordinal",
        "mram_push_ordinal_since_launch", "replay_delay_requested_us",
    )
    require(
        all(row[field] == "" for row in nontransfer_rows for field in transfer_only_fields),
        "lifecycle or launch event carries transfer metadata",
    )

    durations_by_subop = defaultdict(list)
    for row in transfer_rows:
        event_id = int(row["event_id"])
        iteration = int(row["iteration"])
        previous = events[event_id - 1]
        expected = expected_transfer(row["subop"], nr_dpus, rounded, data_logical)
        size = int(expected["size"])
        logical = list(expected["logical"])
        require(row["sdk_api_kind"] == "PUSH_XFER", "API kind differs")
        require(row["timing_scope"] == "PUSH_ONLY", "timing scope differs")
        require(int(row["transfer_bytes_per_dpu"]) == size, "transfer size differs")
        require(int(row["active_dpus"]) == nr_dpus, "active DPUs differ")
        require(int(row["active_ranks"]) == ranks, "active ranks differ")
        require(row["rank_ordinal"] == "ALL", "rank marker differs")
        require(row["dpu_id_in_rank"] == "ALL", "DPU marker differs")
        require(row["cpu_dpu_numa_relation"] == "LOCAL", "NUMA relation differs")
        require(row["dpu_ci_ids"] == "0-7", "CI domain differs")
        require(row["dpu_member_ids"] == "0-7", "member domain differs")
        require(int(row["allocated_dpus"]) == nr_dpus, "allocated DPUs differ")
        require(int(row["allocated_ranks"]) == ranks, "allocated ranks differ")
        require(
            row["previous_sdk_mux_domain_class"] == "COLLECTION",
            "predecessor MUX class differs",
        )
        require(
            row["previous_sdk_op_class"] == previous_sdk_op_class(row),
            "predecessor operation class differs",
        )
        require(
            row["previous_sdk_op"] == previous["op"],
            "predecessor operation differs",
        )
        require(
            row["previous_sdk_subop"]
            == (previous["subop"] if previous["subop"] else "NONE"),
            "predecessor subop differs",
        )
        require(
            row["previous_sdk_target_space"]
            == (previous["target_space"] if previous["target_space"] else "NONE"),
            "predecessor target space differs",
        )
        expected_source_count, expected_target_count = expected_reuse_counts(
            row["subop"], iteration
        )
        require(
            int(row["source_buffer_use_count_before"]) == expected_source_count,
            "source buffer use count differs",
        )
        require(
            int(row["target_region_access_count_before"]) == expected_target_count,
            "target region access count differs",
        )
        require(
            row["source_buffer_reuse_class"]
            == source_buffer_reuse_class(expected_source_count),
            "source buffer reuse class differs",
        )
        require(
            row["target_region_reuse_class"]
            == target_region_reuse_class(expected_target_count),
            "target region reuse class differs",
        )
        require(row["diagnostic_copy_ordinal"] == "NONE", "copy ordinal differs")
        require(int(row["replay_delay_requested_us"]) == 0, "replay delay differs")
        expected_mram_ordinal = (
            1 if row["subop"] in {"input_data", "output_data"} else 0
        )
        require(
            int(row["mram_push_ordinal_since_launch"]) == expected_mram_ordinal,
            "MRAM push ordinal differs",
        )
        for field, value in topology.items():
            require(row[field] == value, "event {} {} differs".format(event_id, field))
        require(row["target_space"] == expected["target_space"], "target space differs")
        require(row["target_symbol"] == expected["target_symbol"], "target symbol differs")
        require(int(row["offset_bytes"]) == expected["offset"], "offset differs")
        require(int(row["size_per_dpu_bytes"]) == size, "per-DPU size differs")
        require(int(row["total_logical_bytes"]) == sum(logical), "logical total differs")
        require(
            int(row["total_transfer_bytes"]) == size * nr_dpus,
            "physical total differs",
        )
        event_details = details_by_event[event_id]
        event_details.sort(key=lambda item: int(item["global_dpu_id"]))
        require(len(event_details) == nr_dpus, "event DPU rows differ")
        require(
            [int(item["logical_bytes"]) for item in event_details] == logical,
            "per-DPU logical distribution differs",
        )
        require(
            {int(item["transfer_bytes"]) for item in event_details} == {size},
            "per-DPU physical bytes differ",
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
        "transfer_median_ns": round(statistics.median(
            duration
            for values in durations_by_subop.values()
            for duration in values
        )),
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
        print("ERROR: {}".format(error), file=sys.stderr)
        return 1
    for summary in summaries:
        print(
            "PASS {path} dpus={configured_dpus} ranks={actual_ranks} "
            "events={event_rows} transfer_rows={transfer_rows} "
            "detail_rows={dpu_detail_rows} median_ns={transfer_median_ns}".format(
                **summary
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
