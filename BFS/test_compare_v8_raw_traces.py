#!/usr/bin/env python3

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from compare_v8_raw_traces import compare
from transport_key import transport_key


class CompareV8RawTracesTest(unittest.TestCase):
    def sample_row(
        self,
        repeat_id: int,
        event_id: int,
        subop: str,
        duration_ns: int,
        previous_bytes: int,
        host_page: int,
    ) -> dict[str, str]:
        row = {
            "run_id": "BFS_64dpu_1tl",
            "repeat_id": str(repeat_id),
            "event_id": str(event_id),
            "configured_dpus": "64",
            "actual_ranks": "1",
            "num_tasklets": "1",
            "op": "dpu_copy_to",
            "subop": subop,
            "phase_class": "INIT",
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
            "physical_dpu_identity": (
                "numa:0/channel:1/rank:0/ci:0/member:0"
            ),
            "host_numa_node": "0",
            "cpu_dpu_numa_relation": "LOCAL",
            "same_source_across_group": "1",
            "previous_sdk_topology_relation": "SAME_DPU",
            "previous_sdk_transfer_bytes": str(previous_bytes),
            "host_buffer_page_offset": str(host_page),
            "measured_ns": str(duration_ns),
            "transport_key": "",
        }
        row["transport_key"] = transport_key(row)
        return row

    def write_trace(
        self, path: Path, repeat_id: int, rows: list[dict[str, str]]
    ) -> None:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def summary_by_model(
        self, summary: list[dict[str, object]]
    ) -> dict[str, dict[str, object]]:
        return {
            str(row["model"]): row
            for row in summary
            if row["scope"] == "ALL"
        }

    def test_previous_bytes_resolves_two_cluster_v8_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for repeat_id in range(1, 21):
                path = root / f"trace_{repeat_id:02d}.csv"
                rows = [
                    self.sample_row(
                        repeat_id,
                        0,
                        "visited_init",
                        100 + repeat_id % 3,
                        1536,
                        0,
                    ),
                    self.sample_row(
                        repeat_id,
                        1,
                        "frontier_init",
                        200 + repeat_id % 3,
                        24576,
                        0,
                    ),
                ]
                self.write_trace(path, repeat_id, rows)
                paths.append(path)

            summary, _, transitions = compare(paths)
            models = self.summary_by_model(summary)
            self.assertEqual(models["v8"]["key_groups"], 1)
            self.assertEqual(models["v8"]["unstable_groups"], 1)
            self.assertEqual(models["v8_prev_bytes"]["key_groups"], 2)
            self.assertEqual(models["v8_prev_bytes"]["stable_groups"], 2)
            self.assertEqual(
                models["v8_prev_bytes"]["loo_coverage_pct"], "100.000000"
            )
            self.assertEqual(
                models["v8_prev_bytes"]["p90_abs_pct_error"], "1.000000"
            )
            self.assertEqual(models["v8_host_page"]["unstable_groups"], 1)
            previous_transition = next(
                row
                for row in transitions
                if row["model"] == "v8_prev_bytes"
            )
            self.assertEqual(previous_transition["outcome"], "resolved")
            page_transition = next(
                row
                for row in transitions
                if row["model"] == "v8_host_page"
            )
            self.assertEqual(page_transition["outcome"], "unresolved")

    def test_unique_host_pages_report_coverage_loss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for repeat_id in range(1, 21):
                path = root / f"trace_{repeat_id:02d}.csv"
                rows = [
                    self.sample_row(
                        repeat_id,
                        0,
                        "visited_init",
                        100 + repeat_id % 3,
                        1536,
                        repeat_id * 8,
                    )
                ]
                self.write_trace(path, repeat_id, rows)
                paths.append(path)

            summary, groups, _ = compare(paths)
            models = self.summary_by_model(summary)
            self.assertEqual(models["v8"]["key_groups"], 1)
            self.assertEqual(models["v8"]["stable_groups"], 1)
            self.assertEqual(models["v8_host_page"]["key_groups"], 20)
            self.assertEqual(models["v8_host_page"]["insufficient_groups"], 20)
            self.assertEqual(
                models["v8_host_page"]["loo_coverage_pct"], "0.000000"
            )
            page_groups = [
                row for row in groups if row["model"] == "v8_host_page"
            ]
            self.assertTrue(
                all(row["status"] == "insufficient" for row in page_groups)
            )


if __name__ == "__main__":
    unittest.main()
