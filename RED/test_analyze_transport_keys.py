#!/usr/bin/env python3

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from analyze_transport_keys import analyze
from transport_key import transport_key


class RedTransportKeyAnalysisTest(unittest.TestCase):
    def sample_row(
        self,
        repeat_id: int,
        duration_ns: int,
        warmup: str = "0",
    ) -> dict[str, str]:
        iteration = "0" if warmup == "1" else "1"
        row = {
            "run_id": "RED_256dpu_1tl",
            "repeat_id": str(repeat_id),
            "event_id": "3",
            "configured_dpus": "256",
            "num_tasklets": "1",
            "op": "dpu_transfer",
            "direction": "TO_DPU",
            "sdk_api_kind": "PUSH_XFER",
            "logical_distribution_class": "PARTITIONED_SCATTER",
            "target_space": "MRAM",
            "transfer_bytes_per_dpu": "204800",
            "active_dpus": "256",
            "active_ranks": "4",
            "active_dpus_per_rank": "64|64|64|64",
            "rank_ordinal": "ALL",
            "dpu_id_in_rank": "ALL",
            "same_source_across_group": "0",
            "phase_class": "ITERATIVE",
            "subop": "input_data",
            "iteration": iteration,
            "warmup": warmup,
            "total_logical_bytes": "52428800",
            "total_transfer_bytes": "52428800",
            "target_symbol": "DPU_MRAM_HEAP_POINTER_NAME",
            "offset_bytes": "0",
            "process_state": "fresh_process",
            "pretrace_warmup_runs": "5",
            "host_numa_node": "0",
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
            for repeat_id, duration in enumerate(
                (100, 101, 99, 102, 98), 1
            ):
                path = root / f"trace_{repeat_id:02d}.csv"
                self.write_rows(path, [self.sample_row(repeat_id, duration)])
                paths.append(path)
            summaries, overview = analyze(paths, 5, 5, 25.0, 25.0)
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["status"], "stable")
            self.assertEqual(overview["stable_transport_key_pct"], "100.000")

    def test_groups_same_key_across_tasklet_configurations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.csv"
            second = root / "second.csv"
            first_row = self.sample_row(1, 100)
            second_row = self.sample_row(2, 101)
            second_row["run_id"] = "RED_256dpu_16tl"
            second_row["num_tasklets"] = "16"
            self.write_rows(first, [first_row])
            self.write_rows(second, [second_row])
            summaries, _ = analyze([first, second], 2, 2, 25.0, 25.0)
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["num_tasklets_values"], "1|16")

    def test_warmup_context_remains_in_same_transport_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for repeat_id in range(1, 6):
                path = root / f"trace_{repeat_id:02d}.csv"
                rows = [
                    self.sample_row(repeat_id, 180_000, "1"),
                    self.sample_row(repeat_id, 100_000, "0"),
                ]
                self.write_rows(path, rows)
                paths.append(path)
            summaries, overview = analyze(paths, 10, 5, 25.0, 25.0)
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["warmup_values"], "0|1")
            self.assertEqual(summaries[0]["status"], "unstable")
            self.assertEqual(overview["stable_transport_key_pct"], "0.000")
            self.assertEqual(
                overview["baseline_12_field_stable_transport_key_pct"],
                "0.000",
            )

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


if __name__ == "__main__":
    unittest.main()
