#!/usr/bin/env python3

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze_transport_keys import analyze
from transport_key import transport_key


class SpmvTransportKeyAnalysisTest(unittest.TestCase):
    def sample_row(
        self, repeat_id: int, duration_ns: int
    ) -> dict[str, str]:
        row = {
            "run_id": "SpMV_256dpu_1tl",
            "repeat_id": str(repeat_id),
            "event_id": "4",
            "configured_dpus": "256",
            "actual_ranks": "4",
            "num_tasklets": "1",
            "op": "dpu_copy_to",
            "direction": "TO_DPU",
            "sdk_api_kind": "SINGLE_COPY",
            "logical_distribution_class": "SHARED_REPLICATION",
            "target_space": "MRAM",
            "transfer_bytes_per_dpu": "115696",
            "active_dpus": "1",
            "active_ranks": "1",
            "active_dpus_per_rank": "1",
            "rank_ordinal": "0",
            "dpu_id_in_rank": "0",
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
            "phase_class": "INIT",
            "subop": "input_vector",
            "global_dpu_id": "0",
            "logical_bytes": "115696",
            "offset_feature": "off=4096:a8=1:a64=1:p4k=1:pages=29",
            "op_call_index": "2",
            "dpu_op_call_index": "2",
            "process_state": "fresh_process",
            "pretrace_warmup_runs": "5",
            "host_numa_node": "0",
            "previous_sdk_topology_relation": "SAME_DPU",
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
            self.assertEqual(overview["stable_transport_key_pct"], "100.000")

    def test_groups_same_transfer_key_across_tasklet_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.csv"
            second = root / "second.csv"
            first_row = self.sample_row(1, 100)
            second_row = self.sample_row(2, 101)
            second_row["run_id"] = "SpMV_256dpu_16tl"
            second_row["num_tasklets"] = "16"
            self.write_rows(first, [first_row])
            self.write_rows(second, [second_row])
            summaries, _ = analyze([first, second], 2, 2, 25.0, 25.0)
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["num_tasklets_values"], "1|16")

    def test_cv_rejects_rare_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for repeat_id in range(1, 21):
                path = root / f"trace_{repeat_id:02d}.csv"
                duration = 10_000 if repeat_id == 20 else 100
                self.write_rows(path, [self.sample_row(repeat_id, duration)])
                paths.append(path)
            summaries, _ = analyze(paths, 20, 20, 25.0, 25.0)
            self.assertEqual(summaries[0]["status"], "unstable")
            self.assertEqual(summaries[0]["status_reason"], "cv>25%")

    def test_rejects_invalid_embedded_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            row = self.sample_row(1, 100)
            row["transport_key"] += ";extra=1"
            self.write_rows(path, [row])
            with self.assertRaisesRegex(ValueError, "invalid transport_key"):
                analyze([path], 2, 2, 25.0, 25.0)


if __name__ == "__main__":
    unittest.main()
