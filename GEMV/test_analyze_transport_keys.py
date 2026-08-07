from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from analyze_transport_keys import analyze
from test_transport_key import transfer_row
from transport_key import transport_key


def analysis_row(measured_ns: int, repeat_id: int) -> dict[str, str]:
    row = transfer_row()
    row.update(
        {
            "event_id": "1",
            "configured_dpus": "128",
            "num_tasklets": "16",
            "subop": "input_vector",
            "phase_class": "WARMUP" if repeat_id % 2 else "ITERATIVE",
            "warmup": "1" if repeat_id % 2 else "0",
            "iteration": "0",
            "total_logical_bytes": str(32768 * 128),
            "total_transfer_bytes": str(32768 * 128),
            "target_symbol": "DPU_MRAM_HEAP_POINTER_NAME",
            "offset_bytes": "0",
            "run_id": "GEMV_128dpu_16tl",
            "repeat_id": str(repeat_id),
            "measured_ns": str(measured_ns),
        }
    )
    row["transport_key"] = transport_key(row)
    return row


class AnalyzeTransportKeysTests(unittest.TestCase):
    def write_trace(self, path: Path, row: dict[str, str]) -> None:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    def test_repeated_low_spread_group_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index, duration in enumerate((100, 102, 101, 99), start=1):
                path = root / f"trace_{index}.csv"
                self.write_trace(path, analysis_row(duration, index))
                paths.append(path)
            summaries, overview = analyze(paths, 4, 4, 10.0, 10.0)
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["status"], "stable")
        self.assertEqual(summaries[0]["phase_classes"], "ITERATIVE|WARMUP")
        self.assertEqual(summaries[0]["warmup_sample_count"], 2)
        self.assertEqual(summaries[0]["iterative_sample_count"], 2)
        self.assertEqual(overview["stable_transport_key_pct"], "100.000")

    def test_large_spread_group_is_unstable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index, duration in enumerate((100, 100, 100, 400), start=1):
                path = root / f"trace_{index}.csv"
                self.write_trace(path, analysis_row(duration, index))
                paths.append(path)
            summaries, _ = analyze(paths, 4, 4, 25.0, 25.0)
        self.assertEqual(summaries[0]["status"], "unstable")


if __name__ == "__main__":
    unittest.main()
