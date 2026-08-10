from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from calibrate_heartbeat_threshold import calibrate, nearest_rank_percentile


class CalibrateHeartbeatThresholdTests(unittest.TestCase):
    def test_uses_minimum_when_idle_p99_is_lower(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "heartbeat.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["lateness_ns"])
                writer.writeheader()
                for index in range(200):
                    writer.writerow({"lateness_ns": 55000 + index})
            summary = calibrate(path, minimum_us=200, percentile=99)
        self.assertEqual(summary["chosen_threshold_us"], 200)

    def test_nearest_rank_percentile(self) -> None:
        self.assertEqual(nearest_rank_percentile(list(range(1, 101)), 99), 99)


if __name__ == "__main__":
    unittest.main()
