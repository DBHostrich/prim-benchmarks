from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from analyze_context_keys import analyze, previous_sdk_op_class
from test_transport_key import transfer_row
from transport_key import transport_key


class AnalyzeContextKeysTests(unittest.TestCase):
    def sample_row(
        self,
        repeat_id: int,
        subop: str,
        active_dpus: int,
        previous_sdk_op: str,
        warmup: int,
        measured_ns: int,
        event_id: int,
    ) -> dict[str, str]:
        row = transfer_row()
        row.update(
            {
                "run_id": f"GEMV_{active_dpus}dpu_16tl",
                "repeat_id": str(repeat_id),
                "event_id": str(event_id),
                "configured_dpus": str(active_dpus),
                "num_tasklets": "16",
                "subop": subop,
                "active_dpus": str(active_dpus),
                "allocated_dpus": str(active_dpus),
                "previous_sdk_op": previous_sdk_op,
                "previous_sdk_direction": (
                    "FROM_DPU"
                    if previous_sdk_op == "dpu_push_xfer"
                    and subop == "input_arguments"
                    else "TO_DPU"
                    if previous_sdk_op == "dpu_push_xfer"
                    else "NONE"
                ),
                "previous_sdk_target_space": (
                    "MRAM" if previous_sdk_op == "dpu_push_xfer" else "NONE"
                ),
                "warmup": str(warmup),
                "iteration": "0" if warmup else "1",
                "source_buffer_reuse_class": (
                    "FIRST_USE" if warmup else "REUSED"
                ),
                "target_region_reuse_class": (
                    "FIRST_ACCESS" if warmup else "REUSED"
                ),
                "measured_ns": str(measured_ns),
            }
        )
        if subop == "input_arguments":
            row.update(
                {
                    "logical_distribution_class": "PARTITIONED_SCATTER",
                    "target_space": "WRAM",
                    "transfer_bytes_per_dpu": "16",
                    "same_source_across_group": "0",
                }
            )
        elif subop == "output_vector":
            row.update(
                {
                    "direction": "FROM_DPU",
                    "logical_distribution_class": "PARTITIONED_GATHER",
                    "transfer_bytes_per_dpu": "32",
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

    def write_schedule(self, path: Path, repeats: range) -> None:
        rows = []
        for repeat_id in repeats:
            for slot, active_dpus in enumerate((256, 1024)):
                rows.append(
                    {
                        "phase": "trace",
                        "round": repeat_id,
                        "slot": slot,
                        "config": f"GEMV_{active_dpus}dpu_16tl",
                        "start_wall_ns": repeat_id * 1000 + slot,
                        "end_wall_ns": repeat_id * 1000 + slot + 1,
                    }
                )
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_previous_sdk_operation_classes(self) -> None:
        self.assertEqual(previous_sdk_op_class({"previous_sdk_op": "dpu_load"}), "LOAD")
        self.assertEqual(
            previous_sdk_op_class(
                {
                    "previous_sdk_op": "dpu_push_xfer",
                    "previous_sdk_direction": "FROM_DPU",
                    "previous_sdk_target_space": "MRAM",
                }
            ),
            "PUSH_XFER_FROM_DPU_MRAM",
        )
        self.assertEqual(
            previous_sdk_op_class({"previous_sdk_op": "dpu_launch"}),
            "LAUNCH_SYNC",
        )

    def test_context_candidate_holdout_and_anomaly_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            repeats = range(1, 6)
            for repeat_id in repeats:
                path = root / f"trace_{repeat_id:02d}.csv"
                rows = [
                    self.sample_row(
                        repeat_id, "input_arguments", 256, "dpu_load", 1,
                        200 + repeat_id, 1,
                    ),
                    self.sample_row(
                        repeat_id, "input_arguments", 256,
                        "dpu_push_xfer", 0, 100 + repeat_id, 2,
                    ),
                    self.sample_row(
                        repeat_id, "input_vector", 256,
                        "dpu_push_xfer", 0,
                        2600 if repeat_id == 5 else 300 + repeat_id * 5, 3,
                    ),
                    self.sample_row(
                        repeat_id, "output_vector", 1024, "dpu_launch", 0,
                        250 + repeat_id, 4,
                    ),
                ]
                self.write_trace(path, rows)
                paths.append(path)
            schedule = root / "schedule.csv"
            self.write_schedule(schedule, repeats)

            outputs = analyze(paths, 2, 2, 25.0, 25.0, schedule)

        groups = outputs["context_group_summary"]
        key_counts = {
            row["model"]: row["table_key_count"] for row in groups
        }
        self.assertEqual(key_counts["base_v8"], 3)
        self.assertEqual(key_counts["warmup_v8"], 4)
        self.assertEqual(key_counts["previous_sdk_op_class_v8"], 4)
        self.assertEqual(key_counts["reuse_context_v8"], 4)
        self.assertEqual(key_counts["base_v9"], 4)

        holdout = outputs["holdout_summary"]
        by_model_scope = {
            (row["model"], row["scope"]): row for row in holdout
        }
        base_load = by_model_scope[("base_v8", "input_arguments:LOAD")]
        context_load = by_model_scope[
            ("previous_sdk_op_class_v8", "input_arguments:LOAD")
        ]
        self.assertEqual(context_load["coverage_pct"], "100.000000")
        self.assertLess(
            float(context_load["p90_abs_pct_error"]),
            float(base_load["p90_abs_pct_error"]),
        )

        vector_events = [
            row
            for row in outputs["anomaly_events"]
            if row["subop"] == "input_vector"
        ]
        self.assertEqual(vector_events[0]["repeat_id"], "5")
        self.assertEqual(vector_events[0]["extreme_tukey_outlier"], 1)
        self.assertEqual(vector_events[0]["schedule_slot"], "0")


if __name__ == "__main__":
    unittest.main()
