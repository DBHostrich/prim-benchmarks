#!/usr/bin/env python3

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from analyze_transport_keys import analyze
from transport_key import (
    transport_key,
    transport_key_with_full_context,
    transport_key_without_physical_rank,
)


class TransportKeyAnalysisTest(unittest.TestCase):
    def sample_row(self, repeat_id: int, duration_ns: int) -> dict[str, str]:
        row = {
            "run_id": "BFS_256dpu_1tl",
            "repeat_id": str(repeat_id),
            "event_id": "0",
            "configured_dpus": "256",
            "num_tasklets": "1",
            "op": "dpu_copy_to",
            "subop": "frontier_broadcast",
            "bfs_level": "2",
            "direction": "TO_DPU",
            "sdk_api_kind": "SINGLE_COPY",
            "logical_distribution_class": "SHARED_REPLICATION",
            "target_space": "MRAM",
            "transfer_bytes_per_dpu": "24576",
            "active_dpus": "1",
            "active_ranks": "1",
            "active_dpus_per_rank": "1",
            "rank_ordinal": "0",
            "dpu_id_in_rank": "0",
            "sdk_physical_rank_id": "12288",
            "sdk_slice_id": "0",
            "sdk_member_id": "0",
            "physical_dpu_identity": "rank:12288/slice:0/member:0",
            "same_source_across_group": "1",
            "phase_class": "ITERATIVE",
            "previous_sdk_op": "dpu_copy_to",
            "previous_sdk_direction": "TO_DPU",
            "previous_sdk_transfer_bytes": "48",
            "previous_sdk_topology_relation": "SAME_RANK",
            "previous_dpu_direction": "FROM_DPU",
            "previous_dpu_transfer_bytes": "24576",
            "previous_dpu_target_relation": "SAME_REGION",
            "launches_since_previous_dpu_transfer": "0",
            "target_region_reuse_class": "REUSED_REGION",
            "host_buffer_page_offset": "0",
            "host_buffer_reuse_class": "SAME_DIRECTION_REUSE",
            "logical_bytes": "24576",
            "offset_feature": "off=0:a8=1:a64=1:p4k=0:pages=6",
            "process_state": "fresh_process",
            "pretrace_warmup_runs": "5",
            "host_numa_node": "0",
            "op_call_index": "0",
            "dpu_op_call_index": "0",
            "transport_key": "",
            "measured_ns": str(duration_ns),
        }
        row["transport_key"] = transport_key(row)
        return row

    def write_rows(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_classifies_tight_repeated_key_as_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for repeat_id, duration in enumerate((100, 101, 99, 102, 98), 1):
                path = root / f"trace_{repeat_id:02d}.csv"
                self.write_rows(path, [self.sample_row(repeat_id, duration)])
                paths.append(path)
            summaries, overview = analyze(paths, 5, 5, 25.0, 25.0)
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["status"], "stable")
            self.assertEqual(summaries[0]["sample_count"], 5)
            self.assertEqual(overview["stable_groups"], 1)
            self.assertEqual(overview["stable_transport_key_pct"], "100.000")
            self.assertEqual(overview["stable_event_pct"], "100.000")

    def test_groups_strictly_by_transport_key_across_configurations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.csv"
            second = root / "second.csv"
            self.write_rows(first, [self.sample_row(1, 100)])
            second_row = self.sample_row(2, 100)
            second_row["configured_dpus"] = "512"
            second_row["num_tasklets"] = "2"
            self.write_rows(second, [second_row])
            summaries, _ = analyze([first, second], 2, 2, 25.0, 25.0)
            self.assertEqual(len(summaries), 1)
            self.assertEqual(
                summaries[0]["num_tasklets_values"], "1|2"
            )
            self.assertEqual(
                summaries[0]["configured_dpus_values"], "256|512"
            )

    def test_accepts_repeated_key_within_one_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            first = self.sample_row(1, 100)
            second = self.sample_row(1, 103)
            second["bfs_level"] = "3"
            self.write_rows(path, [first, second])
            summaries, _ = analyze([path], 2, 2, 25.0, 25.0)
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["sample_count"], 2)
            self.assertEqual(summaries[0]["samples_per_trace_max"], 2)
            self.assertEqual(summaries[0]["bfs_levels"], "2|3")
            self.assertEqual(summaries[0]["status"], "insufficient")
            self.assertEqual(summaries[0]["status_reason"], "traces<2")

    def test_cv_can_reject_a_group_with_tight_central_percentiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for repeat_id in range(1, 21):
                path = root / f"trace_{repeat_id:02d}.csv"
                duration = 10_000 if repeat_id == 20 else 100
                self.write_rows(path, [self.sample_row(repeat_id, duration)])
                paths.append(path)
            summaries, _ = analyze(paths, 20, 20, 25.0, 25.0)
            self.assertEqual(summaries[0]["p90_p10_spread_pct"], "0.000")
            self.assertEqual(summaries[0]["status"], "unstable")
            self.assertEqual(summaries[0]["status_reason"], "cv>25%")

    def test_hardware_context_separates_init_and_iterative_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            iterative = self.sample_row(1, 149_000)
            initial = self.sample_row(1, 89_000)
            initial["subop"] = "frontier_init"
            initial["bfs_level"] = ""
            initial["phase_class"] = "INIT"
            initial["previous_dpu_direction"] = "TO_DPU"
            initial["previous_dpu_target_relation"] = "DIFFERENT_REGION"
            initial["target_region_reuse_class"] = "FIRST_REGION_ACCESS"
            initial["host_buffer_reuse_class"] = "FIRST_SDK_USE"
            initial["transport_key"] = transport_key(initial)
            self.write_rows(path, [initial, iterative])
            summaries, _ = analyze([path], 2, 2, 25.0, 25.0)
            self.assertEqual(len(summaries), 2)
            self.assertEqual(
                {row["phase_class_values"] for row in summaries},
                {"INIT", "ITERATIVE"},
            )

    def test_phase_class_is_diagnostic_outside_v5_key(self) -> None:
        iterative = self.sample_row(1, 100)
        initial = dict(iterative)
        initial["phase_class"] = "INIT"
        self.assertEqual(transport_key(initial), transport_key(iterative))

    def test_physical_rank_identity_separates_v5_key(self) -> None:
        first = self.sample_row(1, 100)
        second = dict(first)
        second["sdk_physical_rank_id"] = "12289"
        second["physical_dpu_identity"] = "rank:12289/slice:0/member:0"
        self.assertNotEqual(transport_key(first), transport_key(second))
        self.assertEqual(
            transport_key_without_physical_rank(first),
            transport_key_without_physical_rank(second),
        )

    def test_reanalyzes_stored_v3_trace_with_v5_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            row = self.sample_row(1, 100)
            row["transport_key"] = transport_key_with_full_context(row)
            row.pop("sdk_physical_rank_id")
            row.pop("physical_dpu_identity")
            self.write_rows(path, [row])
            summaries, overview = analyze([path], 2, 2, 25.0, 25.0)
            self.assertEqual(len(summaries), 1)
            self.assertTrue(str(summaries[0]["transport_key"]).startswith("v5;"))
            self.assertEqual(summaries[0]["sdk_physical_rank_id"], "unknown")
            self.assertEqual(overview["full_context_v3_groups"], 1)

    def test_reports_same_trace_phase_ab_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for repeat_id in range(1, 6):
                path = root / f"trace_{repeat_id:02d}.csv"
                initial = self.sample_row(repeat_id, 89_000)
                initial["subop"] = "frontier_init"
                initial["bfs_level"] = ""
                initial["phase_class"] = "INIT"
                initial["previous_dpu_direction"] = "TO_DPU"
                initial["previous_dpu_target_relation"] = "DIFFERENT_REGION"
                initial["target_region_reuse_class"] = "FIRST_REGION_ACCESS"
                initial["host_buffer_reuse_class"] = "FIRST_SDK_USE"
                initial["transport_key"] = transport_key(initial)
                iterative = self.sample_row(repeat_id, 149_000)
                self.write_rows(path, [initial, iterative])
                paths.append(path)
            summaries, overview = analyze(paths, 5, 5, 25.0, 25.0)
            self.assertEqual(len(summaries), 2)
            self.assertEqual(overview["stable_transport_key_pct"], "100.000")
            self.assertEqual(overview["stable_event_pct"], "100.000")
            self.assertEqual(
                overview["phase_v2_stable_transport_key_pct"], "100.000"
            )
            self.assertEqual(
                overview["baseline_12_field_stable_transport_key_pct"],
                "0.000",
            )
            self.assertEqual(
                overview["baseline_12_field_stable_event_pct"], "0.000"
            )


if __name__ == "__main__":
    unittest.main()
