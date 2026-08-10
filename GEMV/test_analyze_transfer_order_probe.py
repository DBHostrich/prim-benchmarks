from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from analyze_transfer_order_probe import analyze


class AnalyzeTransferOrderProbeTests(unittest.TestCase):
    def write_trace(
        self,
        root: Path,
        order: str,
        repeat: int,
        slow_current: bool,
    ) -> None:
        directory = root / f"GEMV_128dpu_16tl_{order}"
        directory.mkdir(parents=True, exist_ok=True)
        rows = []
        event_id = 0
        for iteration in range(4):
            context = "FIRST_USE" if iteration == 0 else "REUSED"
            predecessor = (
                "input_matrix"
                if order == "MATRIX_THEN_VECTOR"
                else "input_arguments"
            )
            previous_duration = 500 if slow_current else 100
            rows.append(
                {
                    "run_id": directory.name,
                    "repeat_id": str(repeat),
                    "event_id": str(event_id),
                    "op": "dpu_push_xfer",
                    "subop": predecessor,
                    "measured_ns": str(previous_duration),
                    "transfer_order_variant": order,
                }
            )
            event_id += 1
            duration = (
                400
                if order == "MATRIX_THEN_VECTOR"
                and slow_current
                and iteration in {0, 1}
                else 100
            )
            rows.append(
                {
                    "run_id": directory.name,
                    "repeat_id": str(repeat),
                    "event_id": str(event_id),
                    "op": "dpu_push_xfer",
                    "subop": "input_vector",
                    "source_buffer_reuse_class": context,
                    "previous_sdk_subop": predecessor,
                    "previous_sdk_op_class": f"PREVIOUS_{predecessor}",
                    "previous_sdk_transfer_bytes": "100",
                    "transport_key": f"key_{order}_{context}",
                    "measured_ns": str(duration),
                    "thread_cpu_ns": "20",
                    "wall_minus_thread_cpu_ns": str(duration - 20),
                    "cpu_id_start": "0",
                    "cpu_id_end": "0",
                    "voluntary_context_switch_delta": "1",
                    "involuntary_context_switch_delta": "0",
                    "minor_fault_delta": "0",
                    "major_fault_delta": "0",
                    "host_start_ns": str(1000000 + event_id * 1000),
                    "host_end_ns": str(1000000 + event_id * 1000 + duration),
                    "transfer_order_variant": order,
                    "iteration": str(iteration),
                }
            )
            event_id += 1
        trace_path = directory / f"trace_{repeat:02d}.csv"
        fields = sorted({field for row in rows for field in row})
        with trace_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        heartbeat_path = directory / f"heartbeat_{repeat:02d}.csv"
        with heartbeat_path.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "planned_raw_ns",
                    "actual_raw_ns",
                    "actual_mono_ns",
                    "lateness_ns",
                    "cpu_id",
                ],
            )
            writer.writeheader()

    def test_detects_dispersion_and_tail_reduction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for order in ("MATRIX_THEN_VECTOR", "VECTOR_THEN_MATRIX"):
                for repeat in range(1, 13):
                    self.write_trace(
                        root,
                        order,
                        repeat,
                        slow_current=repeat >= 10,
                    )
            outputs = analyze(root)
        comparisons = outputs["transfer_order_comparison"]
        self.assertEqual(len(comparisons), 2)
        self.assertEqual(
            {row["decision"] for row in comparisons},
            {"SUPPORTS_PREDECESSOR_TRANSFER_STATE"},
        )


if __name__ == "__main__":
    unittest.main()
