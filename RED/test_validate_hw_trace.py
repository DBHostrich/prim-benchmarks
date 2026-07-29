#!/usr/bin/env python3

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import validate_hw_trace


EVENT_FIELDS = [
    "run_id",
    "repeat_id",
    "event_id",
    "configured_dpus",
    "actual_ranks",
    "num_tasklets",
    "total_input_elements",
    "total_input_bytes",
    "iteration",
    "warmup",
    "op",
    "subop",
    "direction",
    "active_dpus",
    "size_per_dpu_bytes",
    "total_logical_bytes",
    "total_transfer_bytes",
    "target_space",
    "target_symbol",
    "offset_bytes",
    "host_start_ns",
    "host_end_ns",
    "measured_ns",
]

DPU_FIELDS = [
    "run_id",
    "repeat_id",
    "event_id",
    "configured_dpus",
    "actual_ranks",
    "num_tasklets",
    "iteration",
    "warmup",
    "op",
    "subop",
    "direction",
    "global_dpu_id",
    "rank_ordinal",
    "dpu_id_in_rank",
    "logical_bytes",
    "transfer_bytes",
    "kernel_cycles",
]


class RedTraceValidatorTest(unittest.TestCase):
    def write_trace(
        self, directory: Path, nr_dpus: int, tasklets: int
    ) -> tuple[Path, Path]:
        event_path = directory / "trace_01.csv"
        dpu_path = directory / "trace_01_dpus.csv"
        events: list[dict[str, str]] = []
        details: list[dict[str, str]] = []
        timestamp = 1_000_000
        run_id = f"RED_{nr_dpus}dpu_{tasklets}tl"
        ranks = nr_dpus // 64

        def append(
            op: str,
            subop: str = "",
            direction: str = "",
            iteration: int | None = None,
            warmup: int | None = None,
        ) -> None:
            nonlocal timestamp
            event_id = len(events)
            transfer = op == "dpu_transfer"
            size_per_dpu = ""
            total_bytes = ""
            target_space = ""
            target_symbol = ""
            offset = ""
            if transfer:
                (
                    expected_direction,
                    target_space,
                    target_symbol,
                    byte_count,
                ) = validate_hw_trace.transfer_expectation(
                    subop, nr_dpus, tasklets
                )
                self.assertEqual(direction, expected_direction)
                size_per_dpu = str(byte_count)
                total_bytes = str(byte_count * nr_dpus)
                offset = "0"

            event = {
                "run_id": run_id,
                "repeat_id": "1",
                "event_id": str(event_id),
                "configured_dpus": str(nr_dpus),
                "actual_ranks": str(ranks),
                "num_tasklets": str(tasklets),
                "total_input_elements": str(
                    validate_hw_trace.TOTAL_INPUT_ELEMENTS
                ),
                "total_input_bytes": str(validate_hw_trace.TOTAL_INPUT_BYTES),
                "iteration": "" if iteration is None else str(iteration),
                "warmup": "" if warmup is None else str(warmup),
                "op": op,
                "subop": subop,
                "direction": direction,
                "active_dpus": str(nr_dpus),
                "size_per_dpu_bytes": size_per_dpu,
                "total_logical_bytes": total_bytes,
                "total_transfer_bytes": total_bytes,
                "target_space": target_space,
                "target_symbol": target_symbol,
                "offset_bytes": offset,
                "host_start_ns": str(timestamp),
                "host_end_ns": str(timestamp + 100),
                "measured_ns": "100",
            }
            events.append(event)
            timestamp += 200

            if transfer:
                for dpu_id in range(nr_dpus):
                    details.append(
                        {
                            "run_id": run_id,
                            "repeat_id": "1",
                            "event_id": str(event_id),
                            "configured_dpus": str(nr_dpus),
                            "actual_ranks": str(ranks),
                            "num_tasklets": str(tasklets),
                            "iteration": str(iteration),
                            "warmup": str(warmup),
                            "op": op,
                            "subop": subop,
                            "direction": direction,
                            "global_dpu_id": str(dpu_id),
                            "rank_ordinal": str(dpu_id // 64),
                            "dpu_id_in_rank": str(dpu_id % 64),
                            "logical_bytes": size_per_dpu,
                            "transfer_bytes": size_per_dpu,
                            "kernel_cycles": "",
                        }
                    )

        append("dpu_alloc")
        append("dpu_load")
        for iteration in range(validate_hw_trace.ITERATIONS):
            warmup = 1 if iteration == 0 else 0
            append(
                "dpu_transfer",
                "input_arguments",
                "TO_DPU",
                iteration,
                warmup,
            )
            append(
                "dpu_transfer", "input_data", "TO_DPU", iteration, warmup
            )
            append("dpu_launch", "sync", "", iteration, warmup)
            append(
                "dpu_transfer", "results", "FROM_DPU", iteration, warmup
            )
        append("dpu_free")

        with event_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=EVENT_FIELDS)
            writer.writeheader()
            writer.writerows(events)
        with dpu_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=DPU_FIELDS)
            writer.writeheader()
            writer.writerows(details)
        return event_path, dpu_path

    def test_valid_256_dpu_1_tasklet_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            event_path, dpu_path = self.write_trace(Path(directory), 256, 1)
            summary = validate_hw_trace.validate(event_path, dpu_path)
            self.assertEqual(summary["events"], 19)
            self.assertEqual(summary["dpu_rows"], 3_072)
            self.assertEqual(summary["h2d_transfer_bytes"], 209_731_584)
            self.assertEqual(summary["d2h_transfer_bytes"], 16_384)

    def test_valid_512_dpu_16_tasklet_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            event_path, dpu_path = self.write_trace(Path(directory), 512, 16)
            summary = validate_hw_trace.validate(event_path, dpu_path)
            self.assertEqual(summary["events"], 19)
            self.assertEqual(summary["dpu_rows"], 6_144)
            self.assertEqual(summary["h2d_transfer_bytes"], 209_747_968)
            self.assertEqual(summary["d2h_transfer_bytes"], 524_288)

    def test_rejects_wrong_per_dpu_transfer_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            event_path, dpu_path = self.write_trace(Path(directory), 256, 1)
            with dpu_path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["transfer_bytes"] = "8"
            with dpu_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=DPU_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "per-DPU byte values"):
                validate_hw_trace.validate(event_path, dpu_path)


if __name__ == "__main__":
    unittest.main()
