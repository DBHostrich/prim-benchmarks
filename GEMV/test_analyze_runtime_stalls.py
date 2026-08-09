from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from analyze_runtime_stalls import analyze, heartbeat_overlap, robust_threshold


class AnalyzeRuntimeStallsTests(unittest.TestCase):
    def write_trace(
        self, root: Path, binding: str, repeat: int, measured_ns: int, spike: bool
    ) -> None:
        directory = root / f"GEMV_128dpu_16tl_{binding}"
        directory.mkdir(parents=True, exist_ok=True)
        trace = directory / f"trace_{repeat:02d}.csv"
        heartbeat = directory / f"heartbeat_{repeat:02d}.csv"
        row = {
            "run_id": directory.name,
            "repeat_id": str(repeat),
            "iteration": "1",
            "configured_dpus": "128",
            "active_dpus": "128",
            "host_binding_mode": binding,
            "host_cpu_list": "0" if binding == "FIXED_CORE" else "0,2",
            "op": "dpu_push_xfer",
            "subop": "input_vector",
            "previous_sdk_op_class": "PUSH_XFER_TO_DPU_MRAM",
            "source_buffer_reuse_class": "REUSED",
            "measured_ns": str(measured_ns),
            "thread_cpu_ns": "20",
            "wall_minus_thread_cpu_ns": str(measured_ns - 20),
            "cpu_id_start": "0",
            "cpu_id_end": "0",
            "voluntary_context_switch_delta": "0",
            "involuntary_context_switch_delta": "0",
            "minor_fault_delta": "0",
            "major_fault_delta": "0",
            "host_start_ns": "1000000",
            "host_end_ns": str(1000000 + measured_ns),
            "warmup": "0",
        }
        with trace.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        heartbeat_rows = []
        if spike:
            heartbeat_rows.append(
                {
                    "planned_raw_ns": "1050000",
                    "actual_raw_ns": "1150000",
                    "actual_mono_ns": "2000000",
                    "lateness_ns": "100000",
                    "cpu_id": "4",
                }
            )
        with heartbeat.open("w", newline="") as stream:
            fields = [
                "planned_raw_ns",
                "actual_raw_ns",
                "actual_mono_ns",
                "lateness_ns",
                "cpu_id",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(heartbeat_rows)

    def test_joins_runtime_signal_to_slow_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for repeat, duration in enumerate((100, 100, 100, 500), start=1):
                self.write_trace(
                    root, "MULTI_CORE", repeat, duration, duration == 500
                )
            for repeat in range(1, 5):
                self.write_trace(root, "FIXED_CORE", repeat, 100, False)
            outputs = analyze(root)

        target_events = [
            row
            for row in outputs["runtime_stall_events"]
            if int(row["target"]) == 1
        ]
        slow = [row for row in target_events if int(row["slow_event"]) == 1]
        self.assertEqual(len(slow), 1)
        self.assertEqual(slow[0]["heartbeat_or_preempt_signal"], 1)
        self.assertEqual(slow[0]["heartbeat_spike_count"], 1)
        self.assertEqual(slow[0]["runtime_episode_signal"], 1)
        comparisons = outputs["runtime_stall_binding_comparison"]
        self.assertEqual(len(comparisons), 1)
        self.assertEqual(
            comparisons[0]["decision"], "INSUFFICIENT_BINDING_EVIDENCE"
        )

    def test_robust_threshold_and_interval_overlap(self) -> None:
        self.assertEqual(robust_threshold([100, 100, 100]), 100)
        count, maximum = heartbeat_overlap(
            [(90, 130, 40), (300, 360, 60)], 100, 200, 0
        )
        self.assertEqual((count, maximum), (1, 40))


if __name__ == "__main__":
    unittest.main()
