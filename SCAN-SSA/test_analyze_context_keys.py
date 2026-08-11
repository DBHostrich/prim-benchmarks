from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from analyze_context_keys import analyze
from test_transport_key import transfer_row
from transport_key import previous_sdk_op_class, transport_key


class AnalyzeContextKeysTests(unittest.TestCase):
    def sample_row(
        self,
        repeat_id: int,
        event_id: int,
        subop: str,
        previous_op: str,
        previous_direction: str,
        previous_space: str,
        warmup: int,
        measured_ns: int,
    ) -> dict[str, str]:
        row = transfer_row()
        row.update(
            {
                "run_id": "SCAN_64dpu_16tl",
                "repeat_id": str(repeat_id),
                "event_id": str(event_id),
                "configured_dpus": "64",
                "active_dpus": "64",
                "allocated_dpus": "64",
                "subop": subop,
                "previous_sdk_op": previous_op,
                "previous_sdk_direction": previous_direction,
                "previous_sdk_target_space": previous_space,
                "warmup": str(warmup),
                "iteration": "0" if warmup else "1",
                "source_buffer_reuse_class":
                    "FIRST_USE" if warmup else "REUSED",
                "target_region_reuse_class":
                    "FIRST_ACCESS" if warmup else "REUSED",
                "measured_ns": str(measured_ns),
            }
        )
        if subop == "input_arguments_scan":
            row.update(
                {
                    "logical_distribution_class": "SHARED_REPLICATION",
                    "target_space": "WRAM",
                    "transfer_bytes_per_dpu": "16",
                    "same_source_across_group": "1",
                }
            )
        elif subop == "input_data":
            row.update(
                {
                    "logical_distribution_class": "PARTITIONED_SCATTER",
                    "target_space": "MRAM",
                    "transfer_bytes_per_dpu": "31457280",
                    "same_source_across_group": "0",
                }
            )
        row["previous_sdk_op_class"] = previous_sdk_op_class(row)
        row["transport_key"] = transport_key(row)
        return row

    def write_trace(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_context_models_and_holdout_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for repeat_id in range(1, 6):
                path = root / "trace_{:02d}.csv".format(repeat_id)
                rows = [
                    self.sample_row(
                        repeat_id, 2, "input_arguments_scan", "dpu_load",
                        "NONE", "NONE", 1, 200 + repeat_id,
                    ),
                    self.sample_row(
                        repeat_id, 9, "input_arguments_scan",
                        "dpu_push_xfer", "FROM_DPU", "MRAM", 0,
                        100 + repeat_id,
                    ),
                    self.sample_row(
                        repeat_id, 3, "input_data", "dpu_push_xfer",
                        "TO_DPU", "WRAM", 0,
                        3000 if repeat_id == 5 else 300 + repeat_id,
                    ),
                ]
                self.write_trace(path, rows)
                paths.append(path)
            outputs = analyze(paths, 2, 2, 25.0, 25.0)

        models = {row["model"] for row in outputs["context_group_summary"]}
        self.assertEqual(
            models,
            {
                "base_v8", "warmup_v8", "previous_sdk_op_class_v8",
                "reuse_context_v8", "base_v9",
            },
        )
        key_counts = {
            row["model"]: row["table_key_count"]
            for row in outputs["context_group_summary"]
        }
        self.assertGreater(
            key_counts["previous_sdk_op_class_v8"],
            key_counts["base_v8"],
        )
        holdout = {
            (row["model"], row["scope"]): row
            for row in outputs["holdout_summary"]
        }
        self.assertEqual(
            holdout[
                ("previous_sdk_op_class_v8", "input_arguments_scan:LOAD")
            ]["coverage_pct"],
            "100.000000",
        )
        input_events = [
            row for row in outputs["anomaly_events"]
            if row["subop"] == "input_data"
        ]
        self.assertEqual(input_events[0]["repeat_id"], "5")


if __name__ == "__main__":
    unittest.main()

