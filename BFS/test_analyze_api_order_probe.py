#!/usr/bin/env python3

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from analyze_api_order_probe import (
    CONDITIONS,
    read_and_validate,
    summarize_conditions,
    summarize_pairs,
)
from analyze_group_context_probe import effect_class


FIELDS = [
    "run_id",
    "process_repeat",
    "configured_dpus",
    "num_tasklets",
    "actual_ranks",
    "sample_index",
    "order_index",
    "condition",
    "api_order_class",
    "group_index",
    "group_size",
    "target_global_dpu_id",
    "rank_ordinal",
    "dpu_id_in_rank",
    "sdk_physical_rank_id",
    "sdk_slice_id",
    "sdk_member_id",
    "physical_dpu_identity",
    "rank_boundary_before",
    "same_rank_as_previous_sdk_event",
    "op",
    "direction",
    "sdk_api_kind",
    "logical_distribution_class",
    "target_space",
    "transfer_bytes_per_dpu",
    "active_dpus",
    "active_ranks",
    "active_dpus_per_rank",
    "offset_bytes",
    "same_source_across_group",
    "phase_class",
    "source_buffer_class",
    "same_source_across_conditions",
    "source_pointer",
    "source_content_hash",
    "source_alignment_bytes",
    "target_precondition",
    "previous_op",
    "previous_event_role",
    "previous_direction",
    "previous_transfer_bytes",
    "previous_target_global_dpu_id",
    "same_dpu_as_previous",
    "direction_switched",
    "params_transfer_bytes",
    "interleaved_params",
    "predecessor_d2h_group_ns",
    "source_pretouch_ns",
    "ns_since_previous_sdk_event",
    "sequence_start_ns",
    "sequence_end_ns",
    "sequence_span_ns",
    "host_start_ns",
    "host_end_ns",
    "measured_ns",
    "frontier_sum_ns",
    "verification",
]


def target_order(condition: str) -> list[int]:
    if condition == "PAIR_REVERSED_FRONTIER_PARAMS_PER_DPU":
        return [1, 0, 3, 2]
    return list(range(4))


def previous_fields(
    condition: str, position: int
) -> tuple[str, str, str, int, int, int, int]:
    targets = target_order(condition)
    target = targets[position]
    if condition == "VISITED_FRONTIER_PARAMS_PER_DPU":
        return ("dpu_copy_to", "visited_control", "TO_DPU", 24576, target, 1, 0)
    if position == 0:
        if condition == "D2H_MERGE_FRONTIER_PARAMS_PER_DPU":
            return (
                "dpu_copy_from",
                "frontier_readback",
                "FROM_DPU",
                24576,
                3,
                0,
                1,
            )
        return (
            "dpu_copy_to",
            "frontier_precondition",
            "TO_DPU",
            24576,
            3,
            0,
            0,
        )
    if condition == "CONTIGUOUS_FRONTIER_GROUP":
        return (
            "dpu_copy_to",
            "measured_frontier",
            "TO_DPU",
            24576,
            targets[position - 1],
            0,
            0,
        )
    return (
        "dpu_copy_to",
        "params_control",
        "TO_DPU",
        48,
        targets[position - 1],
        0,
        0,
    )


def make_sequence(process: int, sample: int, order: int) -> list[dict[str, str]]:
    condition = CONDITIONS[(sample + order) % len(CONDITIONS)]
    order_class = {
        "CONTIGUOUS_FRONTIER_GROUP": "CONTIGUOUS_CONTROL",
        "VISITED_FRONTIER_PARAMS_PER_DPU": "INIT_LIKE_ORDER",
        "FRONTIER_PARAMS_PER_DPU": "PARAMS_INTERLEAVED_CONTROL",
        "PAIR_REVERSED_FRONTIER_PARAMS_PER_DPU": "PAIR_REVERSED_CONTROL",
        "D2H_MERGE_FRONTIER_PARAMS_PER_DPU": "ITERATIVE_LIKE_ORDER",
    }[condition]
    interleaved = int(condition != "CONTIGUOUS_FRONTIER_GROUP")
    predecessor_d2h = (
        1000 if condition == "D2H_MERGE_FRONTIER_PARAMS_PER_DPU" else 0
    )
    sequence_start = 1_000_000 + sample * 100_000 + order * 20_000
    cursor = sequence_start + 5
    calls: dict[int, tuple[int, int]] = {}
    targets = target_order(condition)
    for target in targets:
        if condition == "VISITED_FRONTIER_PARAMS_PER_DPU":
            cursor += 50
        if condition == "PAIR_REVERSED_FRONTIER_PARAMS_PER_DPU":
            condition_delta = -20 if target % 2 == 0 else 40
        else:
            condition_delta = {
                "CONTIGUOUS_FRONTIER_GROUP": 0,
                "FRONTIER_PARAMS_PER_DPU": 10,
                "VISITED_FRONTIER_PARAMS_PER_DPU": 30,
                "D2H_MERGE_FRONTIER_PARAMS_PER_DPU": 60,
            }[condition]
        measured = 100 + target + condition_delta + process
        calls[target] = (cursor, cursor + measured)
        cursor += measured
        if interleaved:
            cursor += 10
        else:
            cursor += 3
    sequence_end = cursor + 5
    frontier_sum = sum(end - start for start, end in calls.values())

    rows: list[dict[str, str]] = []
    for position, target in enumerate(targets):
        rank = target // 2
        in_rank = target % 2
        previous_position = position - 1
        boundary = position == 0 or targets[previous_position] // 2 != rank
        (
            previous_op,
            previous_role,
            previous_direction,
            previous_bytes,
            previous_target,
            same_dpu,
            switched,
        ) = previous_fields(condition, position)
        previous_rank = previous_target // 2
        rows.append(
            {
                "run_id": "api_order",
                "process_repeat": str(process),
                "configured_dpus": "4",
                "num_tasklets": "1",
                "actual_ranks": "2",
                "sample_index": str(sample),
                "order_index": str(order),
                "condition": condition,
                "api_order_class": order_class,
                "group_index": str(position),
                "group_size": "4",
                "target_global_dpu_id": str(target),
                "rank_ordinal": str(rank),
                "dpu_id_in_rank": str(in_rank),
                "sdk_physical_rank_id": str(12_288 + rank),
                "sdk_slice_id": "0",
                "sdk_member_id": str(in_rank),
                "physical_dpu_identity": (
                    f"rank:{12_288 + rank}/slice:0/member:{in_rank}"
                ),
                "rank_boundary_before": str(int(boundary)),
                "same_rank_as_previous_sdk_event": str(int(previous_rank == rank)),
                "op": "dpu_copy_to",
                "direction": "TO_DPU",
                "sdk_api_kind": "SINGLE_COPY",
                "logical_distribution_class": "SHARED_REPLICATION",
                "target_space": "MRAM",
                "transfer_bytes_per_dpu": "24576",
                "active_dpus": "1",
                "active_ranks": "1",
                "active_dpus_per_rank": "1",
                "offset_bytes": str(4096 + target * 128),
                "same_source_across_group": "1",
                "phase_class": "CONTROLLED_API_ORDER_PROBE",
                "source_buffer_class": "SHARED_FIXED_BUFFER",
                "same_source_across_conditions": "1",
                "source_pointer": "0x100000",
                "source_content_hash": "0x123456789abcdef0",
                "source_alignment_bytes": "4096",
                "target_precondition": "ZERO_WRITTEN",
                "previous_op": previous_op,
                "previous_event_role": previous_role,
                "previous_direction": previous_direction,
                "previous_transfer_bytes": str(previous_bytes),
                "previous_target_global_dpu_id": str(previous_target),
                "same_dpu_as_previous": str(same_dpu),
                "direction_switched": str(switched),
                "params_transfer_bytes": "48",
                "interleaved_params": str(interleaved),
                "predecessor_d2h_group_ns": str(predecessor_d2h),
                "source_pretouch_ns": "5",
                "ns_since_previous_sdk_event": "2",
                "sequence_start_ns": str(sequence_start),
                "sequence_end_ns": str(sequence_end),
                "sequence_span_ns": str(sequence_end - sequence_start),
                "host_start_ns": str(calls[target][0]),
                "host_end_ns": str(calls[target][1]),
                "measured_ns": str(calls[target][1] - calls[target][0]),
                "frontier_sum_ns": str(frontier_sum),
                "verification": "ok",
            }
        )
    return rows


class ApiOrderProbeAnalysisTest(unittest.TestCase):
    def write_trace(self, root: Path, process: int) -> Path:
        path = root / f"trace_{process}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for sample in range(6):
                for order in range(5):
                    writer.writerows(make_sequence(process, sample, order))
        return path

    def test_balanced_trace_and_order_effects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [self.write_trace(root, process) for process in (1, 2)]
            rows = read_and_validate(paths)
            per_dpu_summaries = summarize_conditions(rows, True)
            group_summaries = summarize_conditions(rows, False)
            per_dpu_pairs = summarize_pairs(rows, True, 5.0)
            group_pairs = summarize_pairs(rows, False, 5.0)

        self.assertEqual(len(rows), 240)
        self.assertEqual(len(per_dpu_summaries), 20)
        self.assertEqual(len(group_summaries), 5)
        self.assertEqual(len(per_dpu_pairs), 20)
        effects = {row["effect"]: row for row in group_pairs}
        self.assertEqual(
            effects["PARAMS_INTERLEAVING_EFFECT"]["effect_class"],
            "CONSISTENT_SLOWER",
        )
        self.assertEqual(
            effects["VISITED_PREDECESSOR_EFFECT"]["effect_class"],
            "CONSISTENT_SLOWER",
        )
        self.assertEqual(
            effects["D2H_GROUP_HISTORY_EFFECT"]["effect_class"],
            "CONSISTENT_SLOWER",
        )
        self.assertEqual(
            effects["ITERATIVE_VS_INIT_ORDER_EFFECT"]["paired_n"], 12
        )
        pair_rows = {
            int(row["target_global_dpu_id"]): row
            for row in per_dpu_pairs
            if row["effect"] == "PAIR_REVERSED_ORDER_EFFECT"
        }
        self.assertEqual(pair_rows[0]["effect_class"], "CONSISTENT_FASTER")
        self.assertEqual(pair_rows[1]["effect_class"], "CONSISTENT_SLOWER")

    def test_previous_event_failure_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_trace(root, 1)
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["previous_event_role"] = "wrong"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "previous-event semantics"):
                read_and_validate([path])

    def test_frontier_sum_failure_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_trace(root, 1)
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows[:4]:
                row["frontier_sum_ns"] = "999"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "frontier sum conservation"):
                read_and_validate([path])

    def test_refined_small_effect_classes(self) -> None:
        self.assertEqual(
            effect_class(3.0, 1.0, 5.0, 5.0),
            "CONSISTENT_SMALL_SLOWER",
        )
        self.assertEqual(
            effect_class(-3.0, -5.0, -1.0, 5.0),
            "CONSISTENT_SMALL_FASTER",
        )
        self.assertEqual(effect_class(0.1, -2.0, 2.0, 5.0), "MIXED")


if __name__ == "__main__":
    unittest.main()
