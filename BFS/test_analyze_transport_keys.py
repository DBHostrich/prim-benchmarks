#!/usr/bin/env python3

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from analyze_transport_keys import analyze
from transport_key import (
    mux_domain_class,
    sdk_topology_relation,
    transport_key,
    transport_key_v7_full,
    transport_key_v6_full,
    transport_key_with_full_context,
    transport_key_with_mux_domain_min,
    transport_key_with_mux_domain_allocated_topology,
    transport_key_with_mux_domain_rank_invariant,
    transport_key_with_mux_relation_min,
    transport_key_with_phase_allocated_topology,
    transport_key_without_mux_pair_context,
    transport_key_without_physical_rank,
)


class TransportKeyAnalysisTest(unittest.TestCase):
    def sample_row(self, repeat_id: int, duration_ns: int) -> dict[str, str]:
        row = {
            "run_id": "BFS_256dpu_1tl",
            "repeat_id": str(repeat_id),
            "event_id": "0",
            "configured_dpus": "256",
            "actual_ranks": "4",
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
            "global_dpu_id": "0",
            "sdk_physical_rank_id": "12288",
            "dpu_sysfs_rank_id": "0",
            "dpu_rank_numa_node": "0",
            "dpu_channel_id": "1",
            "sdk_slice_id": "0",
            "sdk_member_id": "0",
            "dpu_ci_id": "0",
            "dpu_member_id": "0",
            "physical_dpu_identity": "numa:0/channel:1/rank:0/ci:0/member:0",
            "cpu_dpu_numa_relation": "LOCAL",
            "same_source_across_group": "1",
            "phase_class": "ITERATIVE",
            "previous_sdk_op": "dpu_copy_to",
            "previous_sdk_direction": "TO_DPU",
            "previous_sdk_transfer_bytes": "48",
            "previous_sdk_topology_relation": "SAME_MUX_PAIR",
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
            "host_start_ns": "0",
            "host_end_ns": str(duration_ns),
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

    def test_allocated_topology_separates_configurations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.csv"
            second = root / "second.csv"
            self.write_rows(first, [self.sample_row(1, 100)])
            second_row = self.sample_row(2, 100)
            second_row["configured_dpus"] = "512"
            second_row["actual_ranks"] = "8"
            second_row["num_tasklets"] = "2"
            second_row["transport_key"] = transport_key(second_row)
            self.write_rows(second, [second_row])
            summaries, _ = analyze([first, second], 2, 2, 25.0, 25.0)
            self.assertEqual(len(summaries), 2)
            self.assertEqual(
                {row["num_tasklets_values"] for row in summaries},
                {"1", "2"},
            )
            self.assertEqual(
                {row["configured_dpus_values"] for row in summaries},
                {"256", "512"},
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
            initial["previous_sdk_topology_relation"] = "SAME_SLICE"
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

    def test_phase_class_is_diagnostic_outside_v8_key(self) -> None:
        iterative = self.sample_row(1, 100)
        initial = dict(iterative)
        initial["phase_class"] = "INIT"
        self.assertEqual(transport_key(initial), transport_key(iterative))

    def test_physical_rank_identity_separates_v8_key(self) -> None:
        first = self.sample_row(1, 100)
        second = dict(first)
        second["sdk_physical_rank_id"] = "12289"
        second["dpu_sysfs_rank_id"] = "1"
        second["physical_dpu_identity"] = (
            "numa:0/channel:1/rank:1/ci:0/member:0"
        )
        self.assertNotEqual(transport_key(first), transport_key(second))
        self.assertEqual(
            transport_key_without_physical_rank(first),
            transport_key_without_physical_rank(second),
        )

    def test_allocation_local_ordinals_are_outside_v8_key(self) -> None:
        first = self.sample_row(1, 100)
        second = dict(first)
        second["rank_ordinal"] = "7"
        second["dpu_id_in_rank"] = "63"
        second["global_dpu_id"] = "511"
        self.assertEqual(transport_key(first), transport_key(second))
        self.assertNotEqual(
            transport_key_v7_full(first), transport_key_v7_full(second)
        )

    def test_cpu_dpu_numa_relation_separates_v8_key(self) -> None:
        local = self.sample_row(1, 100)
        remote = dict(local)
        remote["host_numa_node"] = "1"
        remote["cpu_dpu_numa_relation"] = "REMOTE"
        self.assertNotEqual(transport_key(local), transport_key(remote))

    def test_sdk_relation_uses_physical_rank_identity(self) -> None:
        previous = self.sample_row(1, 100)
        current = dict(previous)
        current["global_dpu_id"] = "64"
        current["sdk_physical_rank_id"] = "12289"
        current["dpu_sysfs_rank_id"] = "1"
        self.assertEqual(sdk_topology_relation(previous, current), "OTHER_RANK")

    def test_mux_domain_relation_separates_v8_key(self) -> None:
        same_pair = self.sample_row(1, 100)
        other_pair = dict(same_pair)
        other_pair["previous_sdk_topology_relation"] = "SAME_SLICE"
        self.assertNotEqual(transport_key(same_pair), transport_key(other_pair))
        self.assertEqual(
            transport_key_without_mux_pair_context(same_pair),
            transport_key_without_mux_pair_context(other_pair),
        )

    def test_minimal_mux_keys_capture_relation_and_domain(self) -> None:
        same_dpu = self.sample_row(1, 100)
        same_dpu["previous_sdk_topology_relation"] = "SAME_DPU"
        same_pair = dict(same_dpu)
        same_pair["previous_sdk_topology_relation"] = "SAME_MUX_PAIR"
        other_pair = dict(same_dpu)
        other_pair["previous_sdk_topology_relation"] = "SAME_SLICE"

        self.assertNotEqual(
            transport_key_with_mux_relation_min(same_dpu),
            transport_key_with_mux_relation_min(same_pair),
        )
        self.assertEqual(
            transport_key_with_mux_domain_min(same_dpu),
            transport_key_with_mux_domain_min(same_pair),
        )
        self.assertNotEqual(
            transport_key_with_mux_domain_min(same_pair),
            transport_key_with_mux_domain_min(other_pair),
        )
        self.assertEqual(mux_domain_class("COLLECTION"), "COLLECTION")

    def test_rank_invariant_mux_key_shares_rank_local_position(self) -> None:
        first_rank = self.sample_row(1, 100)
        second_rank = dict(first_rank)
        second_rank["rank_ordinal"] = "1"
        second_rank["global_dpu_id"] = "64"
        second_rank["sdk_physical_rank_id"] = "12289"
        second_rank["dpu_sysfs_rank_id"] = "1"
        second_rank["physical_dpu_identity"] = (
            "numa:0/channel:1/rank:1/ci:0/member:0"
        )

        self.assertNotEqual(
            transport_key_with_mux_domain_min(first_rank),
            transport_key_with_mux_domain_min(second_rank),
        )
        self.assertEqual(
            transport_key_with_mux_domain_rank_invariant(first_rank),
            transport_key_with_mux_domain_rank_invariant(second_rank),
        )

    def test_allocated_topology_separates_sdk_allocation_scale(self) -> None:
        small = self.sample_row(1, 100)
        large = dict(small)
        large["configured_dpus"] = "512"
        large["actual_ranks"] = "8"

        self.assertNotEqual(
            transport_key_with_phase_allocated_topology(small),
            transport_key_with_phase_allocated_topology(large),
        )
        self.assertNotEqual(
            transport_key_with_mux_domain_allocated_topology(small),
            transport_key_with_mux_domain_allocated_topology(large),
        )

    def test_reports_minimal_mux_key_phase_mixing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for repeat_id in range(1, 6):
                path = root / f"trace_{repeat_id:02d}.csv"
                initial = self.sample_row(repeat_id, 100)
                initial["phase_class"] = "INIT"
                initial["previous_sdk_topology_relation"] = "SAME_DPU"
                initial["transport_key"] = transport_key(initial)
                iterative = self.sample_row(repeat_id, 102)
                iterative["previous_sdk_topology_relation"] = "SAME_MUX_PAIR"
                iterative["transport_key"] = transport_key(iterative)
                self.write_rows(path, [initial, iterative])
                paths.append(path)

            _, overview = analyze(paths, 5, 5, 25.0, 25.0)
            self.assertEqual(overview["mux_relation_min_groups"], 2)
            self.assertEqual(overview["mux_domain_min_groups"], 1)
            self.assertEqual(
                overview["mux_domain_min_groups_mixing_phase_classes"], 1
            )
            self.assertEqual(
                overview["mux_domain_min_stable_transport_key_pct"],
                "100.000",
            )

    def test_reanalyzes_stored_v3_trace_with_v8_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            row = self.sample_row(1, 100)
            row["transport_key"] = transport_key_with_full_context(row)
            row.pop("sdk_physical_rank_id")
            row.pop("physical_dpu_identity")
            self.write_rows(path, [row])
            summaries, overview = analyze([path], 2, 2, 25.0, 25.0)
            self.assertEqual(len(summaries), 1)
            self.assertTrue(str(summaries[0]["transport_key"]).startswith("v8;"))
            self.assertEqual(summaries[0]["sdk_physical_rank_id"], "unknown")
            self.assertEqual(overview["full_context_v3_groups"], 1)
            self.assertEqual(overview["physical_identity_v5_groups"], 1)

    def test_reanalyzes_stored_v6_trace_with_v8_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            row = self.sample_row(1, 100)
            row["transport_key"] = transport_key_v6_full(row)
            self.write_rows(path, [row])

            summaries, overview = analyze([path], 2, 2, 25.0, 25.0)
            self.assertEqual(len(summaries), 1)
            self.assertTrue(str(summaries[0]["transport_key"]).startswith("v8;"))
            self.assertEqual(
                overview["transport_key_version"],
                "v8_physical_cpu_dpu_topology",
            )

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
                initial["previous_sdk_topology_relation"] = "SAME_SLICE"
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
