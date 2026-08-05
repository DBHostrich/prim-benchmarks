#!/usr/bin/env python3

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from analyze_measurement_labels import analyze


class MeasurementLabelAnalysisTest(unittest.TestCase):
    def write_sample(self, path: Path, repeat_id: int, duration_ns: int) -> None:
        row = {
            "repeat_id": str(repeat_id),
            "configured_dpus": "256",
            "num_tasklets": "1",
            "op": "dpu_copy_to",
            "subop": "frontier_broadcast",
            "direction": "TO_DPU",
            "api_type": "single_copy",
            "logical_distribution_class": "SHARED_REPLICATION",
            "same_source_across_group": "1",
            "target_space": "MRAM",
            "transfer_bytes": "24576",
            "rank_ordinal": "0",
            "dpu_id_in_rank": "0",
            "offset_feature": "off=495144:a8=1:a64=0:p4k=120:pages=7",
            "call_context": (
                "opidx=1536:dpuopidx=6:process=fresh_process:prewarm=5"
            ),
            "host_numa_node": "0",
            "measurement_label": "same-label",
            "measured_ns": str(duration_ns),
        }
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    def test_classifies_tight_repeated_label_as_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for repeat_id, duration in enumerate((100, 101, 99, 102, 98), 1):
                path = root / f"trace_{repeat_id:02d}.csv"
                self.write_sample(path, repeat_id, duration)
                paths.append(path)
            summaries, overview = analyze(paths, 5, 25.0)
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["status"], "stable")
            self.assertEqual(summaries[0]["sample_count"], 5)
            self.assertEqual(overview["stable_groups"], 1)

    def test_keeps_control_configurations_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.csv"
            second = root / "second.csv"
            self.write_sample(first, 1, 100)
            self.write_sample(second, 2, 100)
            with second.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["num_tasklets"] = "2"
            with second.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            summaries, _ = analyze([first, second], 2, 25.0)
            self.assertEqual(len(summaries), 2)
            self.assertEqual(
                {row["num_tasklets"] for row in summaries}, {1, 2}
            )


if __name__ == "__main__":
    unittest.main()
