from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from summarize_scale_stability import summarize


class SummarizeScaleStabilityTests(unittest.TestCase):
    def test_reports_key_and_event_stability_separately(self) -> None:
        rows = [
            {
                "allocated_dpus": "64",
                "allocated_ranks": "1",
                "dpu_sysfs_rank_ids": "0",
                "dpu_channel_ids": "1",
                "sample_count": "30",
                "status": "stable",
                "p90_p10_spread_pct": "2.0",
                "cv_pct": "1.0",
            },
            {
                "allocated_dpus": "64",
                "allocated_ranks": "1",
                "dpu_sysfs_rank_ids": "0",
                "dpu_channel_ids": "1",
                "sample_count": "90",
                "status": "unstable",
                "p90_p10_spread_pct": "40.0",
                "cv_pct": "30.0",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            summary = summarize(path)
        self.assertEqual(summary[0]["status"], "unstable")
        self.assertEqual(summary[0]["stable_key_pct"], "50.000")
        self.assertEqual(summary[0]["stable_event_pct"], "25.000")
        self.assertEqual(summary[0]["stable_samples"], 30)
        self.assertEqual(summary[0]["unstable_samples"], 90)


if __name__ == "__main__":
    unittest.main()
