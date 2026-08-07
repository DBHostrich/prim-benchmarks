from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from summarize_scale_stability import summarize


class SummarizeScaleStabilityTests(unittest.TestCase):
    def test_marks_scale_unstable_when_one_key_is_unstable(self) -> None:
        rows = [
            {
                "allocated_dpus": "64",
                "allocated_ranks": "1",
                "dpu_sysfs_rank_ids": "0",
                "dpu_channel_ids": "1",
                "status": "stable",
                "p90_p10_spread_pct": "2.0",
                "cv_pct": "1.0",
            },
            {
                "allocated_dpus": "64",
                "allocated_ranks": "1",
                "dpu_sysfs_rank_ids": "0",
                "dpu_channel_ids": "1",
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


if __name__ == "__main__":
    unittest.main()
