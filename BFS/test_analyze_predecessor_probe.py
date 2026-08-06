#!/usr/bin/env python3

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from analyze_predecessor_probe import (
    CONDITIONS,
    REQUIRED_FIELDS,
    read_and_validate,
    stability_label,
    summarize_conditions,
    summarize_effects,
    topology_rows,
)


def measured_time(condition: str, target: int, process: int) -> int:
    base = {
        "SAME_DPU_48B_PREDECESSOR": 100,
        "SAME_DPU_24576B_PREDECESSOR": 110,
        "PAIRED_DPU_48B_PREDECESSOR": 160 if target % 2 == 0 else 105,
        "PAIRED_DPU_24576B_PREDECESSOR": 180 if target % 2 == 0 else 125,
    }[condition]
    return base + process


def make_trace(process: int, samples: int = 6) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    configured_dpus = 4
    for sample in range(samples):
        order = list(range(configured_dpus))
        shift = (sample + process) % configured_dpus
        order = order[shift:] + order[:shift]
        rank_positions = {0: 0, 1: 0}
        seed = 1000 + process * 100 + sample
        for visit_position, target in enumerate(order):
            rank = target // 2
            ordinal = target % 2
            call_position = rank_positions[rank]
            rank_positions[rank] += 1
            for condition_order_index in range(len(CONDITIONS)):
                condition = CONDITIONS[
                    (sample + target + condition_order_index) % len(CONDITIONS)
                ]
                same_dpu = condition.startswith("SAME_DPU_")
                predecessor = target if same_dpu else target ^ 1
                predecessor_ordinal = predecessor % 2
                previous_bytes = 24576 if "24576B" in condition else 48
                transition = "SAME_DPU"
                if not same_dpu:
                    transition = "ODD_TO_EVEN" if ordinal == 0 else "EVEN_TO_ODD"
                measured = measured_time(condition, target, process)
                predecessor_start = 30
                predecessor_end = 40
                host_start = 42
                host_end = host_start + measured
                rows.append(
                    {
                        "run_id": "predecessor_factorial",
                        "process_repeat": str(process),
                        "configured_dpus": str(configured_dpus),
                        "num_tasklets": "1",
                        "actual_ranks": "2",
                        "sample_index": str(sample),
                        "target_order_seed": str(seed),
                        "target_visit_position": str(visit_position),
                        "call_position_in_rank": str(call_position),
                        "condition_order_index": str(condition_order_index),
                        "condition": condition,
                        "target_global_dpu_id": str(target),
                        "rank_ordinal": str(rank),
                        "dpu_id_in_rank": str(ordinal),
                        "dpu_ordinal_in_rank": str(ordinal),
                        "sdk_slice_id": str(ordinal),
                        "sdk_member_id": "0",
                        "ordinal_parity": "EVEN" if ordinal == 0 else "ODD",
                        "predecessor_global_dpu_id": str(predecessor),
                        "predecessor_rank_ordinal": str(predecessor // 2),
                        "predecessor_dpu_ordinal_in_rank": str(predecessor_ordinal),
                        "predecessor_slice_id": str(predecessor_ordinal),
                        "predecessor_member_id": "0",
                        "transition_class": transition,
                        "op": "dpu_copy_to",
                        "direction": "TO_DPU",
                        "sdk_api_kind": "SINGLE_COPY",
                        "logical_distribution_class": "SHARED_REPLICATION",
                        "target_space": "MRAM",
                        "transfer_bytes_per_dpu": "24576",
                        "active_dpus": "1",
                        "active_ranks": "1",
                        "active_dpus_per_rank": "1",
                        "offset_bytes": str(1000 + target * 100),
                        "same_source_across_group": "1",
                        "phase_class": "CONTROLLED_PREDECESSOR_FACTORIAL",
                        "source_buffer_class": "SHARED_FIXED_BUFFER",
                        "same_source_across_conditions": "1",
                        "source_pointer": "0x100000",
                        "source_content_hash": "0x123456789abcdef0",
                        "source_alignment_bytes": "4096",
                        "target_precondition": "ZERO_WRITTEN",
                        "previous_op": "dpu_copy_to",
                        "previous_direction": "TO_DPU",
                        "previous_transfer_bytes": str(previous_bytes),
                        "previous_offset_bytes": str(1000 + predecessor * 100),
                        "same_dpu_as_previous": str(int(same_dpu)),
                        "same_rank_as_previous": "1",
                        "precondition_start_ns": "10",
                        "precondition_end_ns": "20",
                        "precondition_ns": "10",
                        "source_pretouch_ns": "5",
                        "predecessor_start_ns": str(predecessor_start),
                        "predecessor_end_ns": str(predecessor_end),
                        "predecessor_ns": str(predecessor_end - predecessor_start),
                        "ns_since_previous_sdk_event": str(host_start - predecessor_end),
                        "host_start_ns": str(host_start),
                        "host_end_ns": str(host_end),
                        "measured_ns": str(measured),
                        "verification": "ok",
                    }
                )
    return rows


def write_trace(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(REQUIRED_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


class PredecessorProbeAnalysisTest(unittest.TestCase):
    def test_factorial_pairing_and_topology(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for process in (1, 2):
                path = Path(directory) / f"probe_{process}.csv"
                write_trace(path, make_trace(process))
                paths.append(path)
            rows = read_and_validate(paths)

        self.assertEqual(len(rows), 2 * 6 * 4 * 4)
        self.assertEqual(len(summarize_conditions(rows, True, 1, 1)), 16)
        self.assertEqual(len(topology_rows(rows)), 4)
        self.assertEqual(
            [row["paired_global_dpu_id"] for row in topology_rows(rows)],
            ["1", "0", "3", "2"],
        )

        parity = summarize_effects(rows, "parity", 5.0)
        locality_48 = {
            row["ordinal_parity"]: row
            for row in parity
            if row["effect"] == "PREDECESSOR_LOCALITY_EFFECT_48B"
        }
        self.assertGreater(float(locality_48["EVEN"]["median_delta_pct"]), 50)
        self.assertLess(float(locality_48["ODD"]["median_delta_pct"]), 6)

    def test_rejects_wrong_predecessor_metadata(self) -> None:
        rows = make_trace(1)
        paired = next(
            row for row in rows if row["condition"].startswith("PAIRED_DPU_")
        )
        paired["predecessor_member_id"] = "7"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            write_trace(path, rows)
            with self.assertRaisesRegex(ValueError, "predecessor topology metadata"):
                read_and_validate([path])

    def test_outlier_sensitive_stability(self) -> None:
        stats, label = stability_label(
            [100] * 99 + [10_000],
            trace_files=5,
            min_samples=20,
            min_traces=5,
            spread_threshold_pct=25.0,
            cv_threshold_pct=25.0,
        )
        self.assertEqual(float(stats["spread_pct"]), 0.0)
        self.assertEqual(label, "OUTLIER_SENSITIVE")


if __name__ == "__main__":
    unittest.main()
