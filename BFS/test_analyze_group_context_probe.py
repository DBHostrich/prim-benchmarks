#!/usr/bin/env python3

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from analyze_group_context_probe import (
    CONDITIONS,
    read_and_validate,
    summarize_group_conditions,
    summarize_group_pairs,
    summarize_per_dpu_conditions,
    summarize_per_dpu_pairs,
)


FIELDS = [
    "run_id",
    "process_repeat",
    "configured_dpus",
    "num_tasklets",
    "actual_ranks",
    "sample_index",
    "order_index",
    "condition",
    "predecessor_chain",
    "group_index",
    "group_size",
    "target_global_dpu_id",
    "rank_ordinal",
    "dpu_id_in_rank",
    "rank_boundary_before",
    "same_rank_as_previous",
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
    "group_previous_op",
    "group_previous_direction",
    "direction_switched",
    "after_launch",
    "predecessor_group_ns",
    "launch_ns",
    "ns_since_previous_sdk_event",
    "source_pretouch_ns",
    "group_start_ns",
    "group_end_ns",
    "host_start_ns",
    "host_end_ns",
    "measured_ns",
    "measured_group_ns",
    "verification",
]


def make_group(process: int, sample: int, order: int) -> list[dict[str, str]]:
    condition = CONDITIONS[(sample + order) % len(CONDITIONS)]
    condition_delta = {
        "H2D_GROUP_THEN_H2D_GROUP": 0,
        "D2H_GROUP_THEN_H2D_GROUP": 20,
        "LAUNCH_D2H_GROUP_THEN_H2D_GROUP": 50,
    }[condition]
    if condition == "H2D_GROUP_THEN_H2D_GROUP":
        previous_op, previous_direction, switched, after_launch, launch_ns = (
            "dpu_copy_to",
            "TO_DPU",
            "0",
            "0",
            "0",
        )
    elif condition == "D2H_GROUP_THEN_H2D_GROUP":
        previous_op, previous_direction, switched, after_launch, launch_ns = (
            "dpu_copy_from",
            "FROM_DPU",
            "1",
            "0",
            "0",
        )
    else:
        previous_op, previous_direction, switched, after_launch, launch_ns = (
            "dpu_copy_from",
            "FROM_DPU",
            "1",
            "1",
            "1000",
        )

    start_base = 1_000_000 + sample * 100_000 + order * 20_000
    rows: list[dict[str, str]] = []
    starts: list[int] = []
    ends: list[int] = []
    cursor = start_base + 5
    for dpu in range(4):
        measured = 100 + dpu + condition_delta + process
        starts.append(cursor)
        ends.append(cursor + measured)
        cursor += measured + 3
    group_start = start_base
    group_end = ends[-1] + 5
    group_ns = group_end - group_start

    for dpu in range(4):
        rank = dpu // 2
        in_rank = dpu % 2
        boundary = dpu == 0 or in_rank == 0
        rows.append(
            {
                "run_id": "group_probe",
                "process_repeat": str(process),
                "configured_dpus": "4",
                "num_tasklets": "1",
                "actual_ranks": "2",
                "sample_index": str(sample),
                "order_index": str(order),
                "condition": condition,
                "predecessor_chain": condition,
                "group_index": str(dpu),
                "group_size": "4",
                "target_global_dpu_id": str(dpu),
                "rank_ordinal": str(rank),
                "dpu_id_in_rank": str(in_rank),
                "rank_boundary_before": str(int(boundary)),
                "same_rank_as_previous": str(int(dpu > 0 and not boundary)),
                "op": "dpu_copy_to",
                "direction": "TO_DPU",
                "sdk_api_kind": "SINGLE_COPY",
                "logical_distribution_class": "SHARED_REPLICATION",
                "target_space": "MRAM",
                "transfer_bytes_per_dpu": "24576",
                "active_dpus": "1",
                "active_ranks": "1",
                "active_dpus_per_rank": "1",
                "offset_bytes": str(4096 + dpu * 128),
                "same_source_across_group": "1",
                "phase_class": "CONTROLLED_GROUP_PROBE",
                "source_buffer_class": "SHARED_FIXED_BUFFER",
                "same_source_across_conditions": "1",
                "source_pointer": "0x100000",
                "source_content_hash": "0x123456789abcdef0",
                "source_alignment_bytes": "4096",
                "target_precondition": "ZERO_WRITTEN",
                "group_previous_op": previous_op,
                "group_previous_direction": previous_direction,
                "direction_switched": switched,
                "after_launch": after_launch,
                "predecessor_group_ns": "500",
                "launch_ns": launch_ns,
                "ns_since_previous_sdk_event": "10",
                "source_pretouch_ns": "5",
                "group_start_ns": str(group_start),
                "group_end_ns": str(group_end),
                "host_start_ns": str(starts[dpu]),
                "host_end_ns": str(ends[dpu]),
                "measured_ns": str(ends[dpu] - starts[dpu]),
                "measured_group_ns": str(group_ns),
                "verification": "ok",
            }
        )
    return rows


class GroupContextProbeAnalysisTest(unittest.TestCase):
    def write_trace(self, root: Path, process: int) -> Path:
        path = root / f"trace_{process}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for sample in range(6):
                for order in range(3):
                    writer.writerows(make_group(process, sample, order))
        return path

    def test_balanced_group_trace_and_effects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [self.write_trace(root, process) for process in (1, 2)]
            rows = read_and_validate(paths)
            per_dpu_summaries = summarize_per_dpu_conditions(rows)
            group_summaries = summarize_group_conditions(rows)
            per_dpu_pairs = summarize_per_dpu_pairs(rows, 5.0)
            group_pairs = summarize_group_pairs(rows, 5.0)

        self.assertEqual(len(rows), 144)
        self.assertEqual(len(per_dpu_summaries), 12)
        self.assertEqual(len(group_summaries), 3)
        self.assertEqual(len(per_dpu_pairs), 12)
        effects = {row["effect"]: row for row in group_pairs}
        self.assertEqual(
            effects["D2H_GROUP_HISTORY_EFFECT"]["effect_class"],
            "CONSISTENT_SLOWER",
        )
        self.assertEqual(
            effects["AFTER_LAUNCH_GROUP_EFFECT"]["effect_class"],
            "CONSISTENT_SLOWER",
        )
        self.assertEqual(
            effects["COMBINED_ITERATIVE_GROUP_EFFECT"]["paired_n"], 12
        )

    def test_group_index_failure_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_trace(root, 1)
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["group_index"] = "2"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "group indexes"):
                read_and_validate([path])

    def test_group_time_conservation_failure_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_trace(root, 1)
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows[:4]:
                row["measured_group_ns"] = "999"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "group time conservation"):
                read_and_validate([path])


if __name__ == "__main__":
    unittest.main()
