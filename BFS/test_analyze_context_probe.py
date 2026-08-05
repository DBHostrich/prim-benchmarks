#!/usr/bin/env python3

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from analyze_context_probe import (
    CONDITIONS,
    read_and_validate,
    summarize_conditions,
    summarize_pairs,
)


FIELDS = [
    "run_id",
    "process_repeat",
    "configured_dpus",
    "num_tasklets",
    "actual_ranks",
    "target_global_dpu_id",
    "rank_ordinal",
    "dpu_id_in_rank",
    "sample_index",
    "order_index",
    "condition",
    "predecessor_chain",
    "op",
    "direction",
    "sdk_api_kind",
    "target_space",
    "transfer_bytes_per_dpu",
    "offset_bytes",
    "source_buffer_class",
    "same_source_across_conditions",
    "previous_op",
    "previous_direction",
    "direction_switched",
    "after_launch",
    "predecessor_ns",
    "launch_ns",
    "ns_since_previous_sdk_event",
    "source_pretouch_ns",
    "host_start_ns",
    "host_end_ns",
    "measured_ns",
    "verification",
]


def make_row(process: int, sample: int, order: int) -> dict[str, str]:
    condition = CONDITIONS[(sample + order) % len(CONDITIONS)]
    measured = {
        "COPY_TO_THEN_COPY_TO": 100,
        "COPY_FROM_THEN_COPY_TO": 120,
        "LAUNCH_COPY_FROM_THEN_COPY_TO": 150,
    }[condition] + process
    if condition == "COPY_TO_THEN_COPY_TO":
        previous_op, previous_direction, switched, after_launch, launch_ns = (
            "dpu_copy_to",
            "TO_DPU",
            "0",
            "0",
            "0",
        )
    elif condition == "COPY_FROM_THEN_COPY_TO":
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
    start = 10_000 + sample * 1_000 + order * 200
    return {
        "run_id": "probe",
        "process_repeat": str(process),
        "configured_dpus": "256",
        "num_tasklets": "1",
        "actual_ranks": "4",
        "target_global_dpu_id": "0",
        "rank_ordinal": "0",
        "dpu_id_in_rank": "0",
        "sample_index": str(sample),
        "order_index": str(order),
        "condition": condition,
        "predecessor_chain": condition,
        "op": "dpu_copy_to",
        "direction": "TO_DPU",
        "sdk_api_kind": "SINGLE_COPY",
        "target_space": "MRAM",
        "transfer_bytes_per_dpu": "24576",
        "offset_bytes": "4096",
        "source_buffer_class": "SHARED_FIXED_BUFFER",
        "same_source_across_conditions": "1",
        "previous_op": previous_op,
        "previous_direction": previous_direction,
        "direction_switched": switched,
        "after_launch": after_launch,
        "predecessor_ns": "50",
        "launch_ns": launch_ns,
        "ns_since_previous_sdk_event": "10",
        "source_pretouch_ns": "5",
        "host_start_ns": str(start),
        "host_end_ns": str(start + measured),
        "measured_ns": str(measured),
        "verification": "ok",
    }


class ContextProbeAnalysisTest(unittest.TestCase):
    def write_trace(self, root: Path, process: int) -> Path:
        path = root / f"trace_{process}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for sample in range(6):
                for order in range(3):
                    writer.writerow(make_row(process, sample, order))
        return path

    def test_balanced_trace_and_paired_effects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [self.write_trace(root, process) for process in (1, 2)]
            rows = read_and_validate(paths)
            summaries = summarize_conditions(rows)
            paired = summarize_pairs(rows, 5.0)

        self.assertEqual(len(rows), 36)
        self.assertEqual(len(summaries), 3)
        effects = {row["effect"]: row for row in paired}
        self.assertEqual(
            effects["DIRECTION_SWITCH_EFFECT"]["effect_class"],
            "CONSISTENT_SLOWER",
        )
        self.assertEqual(
            effects["AFTER_LAUNCH_EFFECT"]["effect_class"],
            "CONSISTENT_SLOWER",
        )
        self.assertEqual(
            effects["COMBINED_ITERATIVE_HISTORY_EFFECT"]["paired_n"], 12
        )

    def test_time_conservation_failure_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_trace(root, 1)
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["measured_ns"] = "999"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "time conservation"):
                read_and_validate([path])


if __name__ == "__main__":
    unittest.main()
