#!/usr/bin/env python3
"""Validate RED hardware event and per-DPU traces."""

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
    same_source_across_group,
    sdk_api_kind,
    transport_key,
)


TOTAL_INPUT_ELEMENTS = 6_553_600
TOTAL_INPUT_BYTES = 52_428_800
ITERATIONS = 4
SUPPORTED_DPUS = {256: 4, 512: 8}
SUPPORTED_TASKLETS = {1, 2, 4, 8, 16}

EVENT_FIELDS = {
    "run_id",
    "repeat_id",
    "event_id",
    "configured_dpus",
    "actual_ranks",
    "num_tasklets",
    "total_input_elements",
    "total_input_bytes",
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
    "same_source_across_group",
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
    "pretrace_warmup_runs",
    "host_numa_node",
    "transport_key",
    "host_start_ns",
    "host_end_ns",
    "measured_ns",
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
    "logical_bytes",
    "transfer_bytes",
    "kernel_cycles",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(path: Path, expected_fields: set[str], kind: str) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        require(
            fields == expected_fields,
            f"{kind} header fields differ: got {sorted(fields)}",
        )
        rows = list(reader)
    require(bool(rows), f"{kind} trace contains zero data rows")
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
    require(
        event_path.name.endswith(".csv"),
        f"event trace path must end with .csv: {event_path}",
    )
    return event_path.with_name(f"{event_path.stem}_dpus.csv")


def expected_sequence() -> list[tuple[str, str, str, str, str]]:
    sequence = [
        ("dpu_alloc", "", "", "", ""),
        ("dpu_load", "", "", "", ""),
    ]
    for iteration in range(ITERATIONS):
        warmup = "1" if iteration == 0 else "0"
        sequence.extend(
            [
                (
                    "dpu_transfer",
                    "input_arguments",
                    "TO_DPU",
                    str(iteration),
                    warmup,
                ),
                (
                    "dpu_transfer",
                    "input_data",
                    "TO_DPU",
                    str(iteration),
                    warmup,
                ),
                ("dpu_launch", "sync", "", str(iteration), warmup),
                (
                    "dpu_transfer",
                    "results",
                    "FROM_DPU",
                    str(iteration),
                    warmup,
                ),
            ]
        )
    sequence.append(("dpu_free", "", "", "", ""))
    return sequence


def transfer_expectation(
    subop: str, nr_dpus: int, tasklets: int
) -> tuple[str, str, str, int]:
    if subop == "input_arguments":
        return "TO_DPU", "WRAM", "DPU_INPUT_ARGUMENTS", 16
    if subop == "input_data":
        return (
            "TO_DPU",
            "MRAM",
            "DPU_MRAM_HEAP_POINTER_NAME",
            TOTAL_INPUT_BYTES // nr_dpus,
        )
    if subop == "results":
        return "FROM_DPU", "WRAM", "DPU_RESULTS", 16 * tasklets
    raise ValueError(f"unknown RED transfer subop={subop}")


def validate(event_path: Path, dpu_path: Path | None = None) -> dict[str, object]:
    if dpu_path is None:
        dpu_path = detail_path_for(event_path)

    events = read_csv(event_path, EVENT_FIELDS, "event")
    details = read_csv(dpu_path, DPU_FIELDS, "DPU")

    nr_dpus = one_int(events, "configured_dpus")
    ranks = one_int(events, "actual_ranks")
    tasklets = one_int(events, "num_tasklets")
    run_id = one_text(events, "run_id")
    repeat_id = one_int(events, "repeat_id")
    process_state = one_text(events, "process_state")
    pretrace_warmup_runs = one_int(events, "pretrace_warmup_runs")
    host_numa_node = one_text(events, "host_numa_node")
    rank_shape = "|".join("64" for _ in range(ranks))

    require(nr_dpus in SUPPORTED_DPUS, f"unsupported configured_dpus={nr_dpus}")
    require(
        ranks == SUPPORTED_DPUS[nr_dpus],
        f"actual_ranks={ranks}, expected {SUPPORTED_DPUS[nr_dpus]}",
    )
    require(
        tasklets in SUPPORTED_TASKLETS,
        f"unsupported num_tasklets={tasklets}",
    )
    require(
        one_int(events, "total_input_elements") == TOTAL_INPUT_ELEMENTS,
        f"total_input_elements differs from {TOTAL_INPUT_ELEMENTS}",
    )
    require(
        one_int(events, "total_input_bytes") == TOTAL_INPUT_BYTES,
        f"total_input_bytes differs from {TOTAL_INPUT_BYTES}",
    )
    require(
        bool(process_state), "process_state metadata is empty"
    )
    require(
        pretrace_warmup_runs >= 0,
        "pretrace_warmup_runs metadata is negative",
    )
    require(
        bool(host_numa_node), "host_numa_node metadata is empty"
    )

    wanted_sequence = expected_sequence()
    require(
        len(events) == len(wanted_sequence),
        f"event rows={len(events)}, expected {len(wanted_sequence)}",
    )
    event_ids = [int(row["event_id"]) for row in events]
    require(
        event_ids == list(range(len(events))),
        "event_id sequence differs from contiguous zero-based IDs",
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
    for event_id, (actual, wanted) in enumerate(
        zip(actual_sequence, wanted_sequence)
    ):
        require(
            actual == wanted,
            f"event {event_id} semantic tuple={actual}, expected {wanted}",
        )

    for row in events:
        event_id = int(row["event_id"])
        start = int(row["host_start_ns"])
        end = int(row["host_end_ns"])
        measured = int(row["measured_ns"])
        require(end >= start, f"event {event_id} has negative elapsed time")
        require(
            measured == end - start,
            f"event {event_id} measured_ns differs from end-start",
        )
        require(measured > 0, f"event {event_id} measured_ns is zero")
        require(
            int(row["active_dpus"]) == nr_dpus,
            f"event {event_id} active_dpus differs from {nr_dpus}",
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
        require(
            row["transport_key"] == transport_key(row),
            f"event {event_id} has invalid transport_key",
        )

    transfer_rows = [row for row in events if row["op"] == "dpu_transfer"]
    collection_rows = [
        row for row in events if row["op"] in {"dpu_alloc", "dpu_load", "dpu_launch", "dpu_free"}
    ]
    empty_transfer_fields = (
        "sdk_api_kind",
        "logical_distribution_class",
        "target_space",
        "transfer_bytes_per_dpu",
        "active_ranks",
        "active_dpus_per_rank",
        "rank_ordinal",
        "dpu_id_in_rank",
        "same_source_across_group",
        "phase_class",
        "size_per_dpu_bytes",
        "total_logical_bytes",
        "total_transfer_bytes",
        "target_symbol",
        "offset_bytes",
        "transport_key",
    )
    require(
        all(
            row[field] == ""
            for row in collection_rows
            for field in empty_transfer_fields
        ),
        "collection event carries transfer metadata",
    )

    h2d_bytes = 0
    d2h_bytes = 0
    for row in transfer_rows:
        event_id = int(row["event_id"])
        direction, target_space, target_symbol, bytes_per_dpu = (
            transfer_expectation(row["subop"], nr_dpus, tasklets)
        )
        total_bytes = bytes_per_dpu * nr_dpus
        require(
            row["direction"] == direction,
            f"event {event_id} direction={row['direction']}, expected {direction}",
        )
        require(
            row["target_space"] == target_space,
            f"event {event_id} target_space={row['target_space']}, expected {target_space}",
        )
        require(
            row["target_symbol"] == target_symbol,
            f"event {event_id} target_symbol={row['target_symbol']}, expected {target_symbol}",
        )
        require(row["offset_bytes"] == "0", f"event {event_id} offset differs from zero")
        require(
            row["sdk_api_kind"] == "PUSH_XFER",
            f"event {event_id} sdk_api_kind differs from PUSH_XFER",
        )
        require(
            row["logical_distribution_class"]
            == (
                "REDUCTION_GATHER"
                if row["subop"] == "results"
                else "PARTITIONED_SCATTER"
            ),
            f"event {event_id} distribution class is inconsistent",
        )
        require(
            int(row["transfer_bytes_per_dpu"]) == bytes_per_dpu,
            f"event {event_id} transfer_bytes_per_dpu differs from {bytes_per_dpu}",
        )
        require(
            int(row["active_ranks"]) == ranks,
            f"event {event_id} active_ranks differs from {ranks}",
        )
        require(
            row["active_dpus_per_rank"] == rank_shape,
            f"event {event_id} active_dpus_per_rank differs from {rank_shape}",
        )
        require(
            row["rank_ordinal"] == "ALL"
            and row["dpu_id_in_rank"] == "ALL",
            f"event {event_id} collection topology marker differs from ALL",
        )
        require(
            row["same_source_across_group"] == "0",
            f"event {event_id} same_source_across_group differs from zero",
        )
        require(
            row["phase_class"] == "ITERATIVE",
            f"event {event_id} phase_class is inconsistent",
        )
        require(
            int(row["size_per_dpu_bytes"]) == bytes_per_dpu,
            f"event {event_id} size_per_dpu_bytes differs from {bytes_per_dpu}",
        )
        require(
            int(row["total_logical_bytes"]) == total_bytes,
            f"event {event_id} total_logical_bytes differs from {total_bytes}",
        )
        require(
            int(row["total_transfer_bytes"]) == total_bytes,
            f"event {event_id} total_transfer_bytes differs from {total_bytes}",
        )
        if direction == "TO_DPU":
            h2d_bytes += total_bytes
        else:
            d2h_bytes += total_bytes

    require(
        len(details) == 12 * nr_dpus,
        f"DPU detail rows={len(details)}, expected {12 * nr_dpus}",
    )
    for field, expected in (
        ("configured_dpus", nr_dpus),
        ("actual_ranks", ranks),
        ("num_tasklets", tasklets),
        ("repeat_id", repeat_id),
    ):
        require(
            one_int(details, field) == expected,
            f"DPU detail {field} differs from event trace",
        )
    require(
        one_text(details, "run_id") == run_id,
        "DPU detail run_id differs from event trace",
    )

    details_by_event: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in details:
        event_id = int(row["event_id"])
        require(
            0 <= event_id < len(events),
            f"DPU detail event_id={event_id} is outside event trace",
        )
        details_by_event[event_id].append(row)
        require(
            row["kernel_cycles"] == "",
            f"DPU detail event {event_id} carries kernel_cycles with PERF=0",
        )

    transfer_event_ids = {int(row["event_id"]) for row in transfer_rows}
    require(
        set(details_by_event) == transfer_event_ids,
        "DPU detail event IDs differ from transfer event IDs",
    )
    topology: dict[int, tuple[int, int]] = {}
    for event in transfer_rows:
        event_id = int(event["event_id"])
        rows = details_by_event[event_id]
        require(
            len(rows) == nr_dpus,
            f"event {event_id} DPU rows={len(rows)}, expected {nr_dpus}",
        )
        dpu_ids = [int(row["global_dpu_id"]) for row in rows]
        require(
            dpu_ids == list(range(nr_dpus)),
            f"event {event_id} global_dpu_id sequence differs from 0..{nr_dpus - 1}",
        )
        for row in rows:
            dpu_id = int(row["global_dpu_id"])
            rank = int(row["rank_ordinal"])
            dpu_in_rank = int(row["dpu_id_in_rank"])
            pair = (rank, dpu_in_rank)
            if dpu_id in topology:
                require(
                    topology[dpu_id] == pair,
                    f"DPU {dpu_id} topology changes across events",
                )
            else:
                topology[dpu_id] = pair
            require(
                0 <= rank < ranks,
                f"DPU {dpu_id} rank_ordinal={rank} is outside actual_ranks",
            )
            require(
                0 <= dpu_in_rank < 64,
                f"DPU {dpu_id} dpu_id_in_rank={dpu_in_rank} is outside 0..63",
            )
            for field in ("iteration", "warmup", "op", "subop", "direction"):
                require(
                    row[field] == event[field],
                    f"event {event_id} DPU detail {field} differs from event row",
                )
        require(
            len({topology[int(row["global_dpu_id"])] for row in rows}) == nr_dpus,
            f"event {event_id} contains duplicate topology coordinates",
        )
        expected_per_dpu = int(event["size_per_dpu_bytes"])
        require(
            all(
                int(row["logical_bytes"]) == expected_per_dpu
                and int(row["transfer_bytes"]) == expected_per_dpu
                for row in rows
            ),
            f"event {event_id} per-DPU byte values differ from {expected_per_dpu}",
        )
        require(
            sum(int(row["logical_bytes"]) for row in rows)
            == int(event["total_logical_bytes"]),
            f"event {event_id} DPU logical-byte sum differs from event total",
        )
        require(
            sum(int(row["transfer_bytes"]) for row in rows)
            == int(event["total_transfer_bytes"]),
            f"event {event_id} DPU transfer-byte sum differs from event total",
        )

    rank_members = Counter(rank for rank, _ in topology.values())
    require(
        rank_members == Counter({rank: 64 for rank in range(ranks)}),
        f"rank membership differs from 64 DPUs per rank: {dict(rank_members)}",
    )

    expected_h2d = ITERATIONS * (TOTAL_INPUT_BYTES + 16 * nr_dpus)
    expected_d2h = ITERATIONS * nr_dpus * 16 * tasklets
    require(
        h2d_bytes == expected_h2d,
        f"H2D transfer bytes={h2d_bytes}, expected {expected_h2d}",
    )
    require(
        d2h_bytes == expected_d2h,
        f"D2H transfer bytes={d2h_bytes}, expected {expected_d2h}",
    )

    durations: dict[str, list[int]] = defaultdict(list)
    for row in events:
        key = row["op"]
        if row["subop"]:
            key = f"{key}:{row['subop']}"
        durations[key].append(int(row["measured_ns"]))

    return {
        "path": str(event_path),
        "dpu_path": str(dpu_path),
        "configured_dpus": nr_dpus,
        "actual_ranks": ranks,
        "num_tasklets": tasklets,
        "events": len(events),
        "dpu_rows": len(details),
        "h2d_transfer_bytes": h2d_bytes,
        "d2h_transfer_bytes": d2h_bytes,
        "durations": durations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate one or more RED trace_*.csv event files."
    )
    parser.add_argument("traces", nargs="+", type=Path)
    args = parser.parse_args()

    all_durations: dict[str, list[int]] = defaultdict(list)
    try:
        for path in args.traces:
            summary = validate(path)
            for key, values in summary["durations"].items():
                all_durations[key].extend(values)
            print(
                f"PASS {path}: DPU={summary['configured_dpus']} "
                f"TL={summary['num_tasklets']} ranks={summary['actual_ranks']} "
                f"events={summary['events']} dpu_rows={summary['dpu_rows']} "
                f"H2D={summary['h2d_transfer_bytes']} "
                f"D2H={summary['d2h_transfer_bytes']}"
            )
    except (OSError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    for key in sorted(all_durations):
        print(
            f"{key.replace(':', '_')}_measured_ns_median="
            f"{int(statistics.median(all_durations[key]))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
