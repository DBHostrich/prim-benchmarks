from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from transport_key import (
    logical_distribution_class,
    phase_class,
    previous_sdk_op_class,
    same_source_across_group,
    sdk_api_kind,
    source_buffer_reuse_class,
    target_region_reuse_class,
    transport_key,
)
from validate_hw_trace import (
    DPU_FIELDS,
    EVENT_FIELDS,
    expected_sequence,
    expected_transfer,
    validate,
)


class ValidateHardwareTraceTests(unittest.TestCase):
    def test_c_trace_uses_canonical_mram_target_label(self) -> None:
        source_path = Path(__file__).with_name("host") / "app.c"
        source = source_path.read_text()
        self.assertEqual(
            source.count('"MRAM", "DPU_MRAM_HEAP_POINTER_NAME"'),
            3,
        )

    def create_trace(
        self,
        root: Path,
        transfer_order_variant: str = "MATRIX_THEN_VECTOR",
        vector_replay_mode: str = "NONE",
    ) -> Path:
        event_path = root / "trace_01.csv"
        detail_path = root / "trace_01_dpus.csv"
        events: list[dict[str, str]] = []
        details: list[dict[str, str]] = []
        previous: dict[str, str] | None = None
        vector_copies_seen: dict[int, int] = {}
        mram_pushes_seen: dict[int, int] = {}
        for event_id, semantic in enumerate(
            expected_sequence(transfer_order_variant, vector_replay_mode)
        ):
            op, subop, direction, iteration, warmup = semantic
            row = {field: "" for field in EVENT_FIELDS}
            row.update(
                {
                    "run_id": "GEMV_64dpu_16tl",
                    "repeat_id": "1",
                    "event_id": str(event_id),
                    "configured_dpus": "64",
                    "actual_ranks": "1",
                    "num_tasklets": "16",
                    "m_size": "8192",
                    "n_size": "8192",
                    "n_size_pad": "8192",
                    "max_rows_per_dpu": "128",
                    "op": op,
                    "subop": subop,
                    "direction": direction,
                    "iteration": iteration,
                    "warmup": warmup,
                    "host_numa_node": "0",
                    "process_state": "interleaved_fresh_process",
                    "host_binding_mode": "FIXED_CORE",
                    "host_cpu_list": "2",
                    "transfer_order_variant": transfer_order_variant,
                    "vector_replay_mode": vector_replay_mode,
                    "pretrace_warmup_runs": "3",
                    "host_start_ns": str(event_id * 100 + 1),
                    "host_end_ns": str(event_id * 100 + 51),
                    "measured_ns": "50",
                    "thread_cpu_ns": "40",
                    "wall_minus_thread_cpu_ns": "10",
                    "cpu_id_start": "2",
                    "cpu_id_end": "2",
                    "voluntary_context_switch_delta": "0",
                    "involuntary_context_switch_delta": "0",
                    "minor_fault_delta": "0",
                    "major_fault_delta": "0",
                }
            )
            if op == "dpu_push_xfer":
                iteration_number = int(iteration)
                expectation = expected_transfer(subop, 64, 128, 8192)
                size = int(expectation["size"])
                logical = list(expectation["logical"])
                if direction == "TO_DPU" and expectation["target_space"] == "MRAM":
                    mram_pushes_seen[iteration_number] = (
                        mram_pushes_seen.get(iteration_number, 0) + 1
                    )
                    mram_ordinal = mram_pushes_seen[iteration_number]
                else:
                    mram_ordinal = 0
                if subop == "input_vector":
                    copy_index = vector_copies_seen.get(iteration_number, 0)
                    vector_copies_seen[iteration_number] = copy_index + 1
                    copy_ordinal = (
                        "PRIMARY" if copy_index == 0 else "IDENTICAL_REPLAY"
                    )
                    use_count = (
                        2 * iteration_number + copy_index
                        if vector_replay_mode == "IDENTICAL_REPLAY"
                        else iteration_number
                    )
                else:
                    copy_ordinal = "NONE"
                    use_count = iteration_number
                row.update(
                    {
                        "sdk_api_kind": sdk_api_kind(op),
                        "timing_scope": "PUSH_ONLY",
                        "logical_distribution_class": logical_distribution_class(
                            op, subop
                        ),
                        "target_space": str(expectation["target_space"]),
                        "transfer_bytes_per_dpu": str(size),
                        "active_dpus": "64",
                        "active_ranks": "1",
                        "active_dpus_per_rank": "64",
                        "rank_ordinal": "ALL",
                        "dpu_id_in_rank": "ALL",
                        "same_source_across_group": same_source_across_group(
                            op, subop
                        ),
                        "dpu_rank_numa_nodes": "0",
                        "cpu_dpu_numa_relation": "LOCAL",
                        "dpu_channel_ids": "1",
                        "dpu_sysfs_rank_ids": "0",
                        "sdk_physical_rank_ids": "12288",
                        "dpu_ci_ids": "0-7",
                        "dpu_member_ids": "0-7",
                        "allocated_topology_signature": "r0@n0@c1",
                        "allocated_dpus": "64",
                        "allocated_ranks": "1",
                        "previous_sdk_op": "NONE" if previous is None else previous["op"],
                        "previous_sdk_direction": (
                            "NONE"
                            if previous is None or previous["direction"] == ""
                            else previous["direction"]
                        ),
                        "previous_sdk_transfer_bytes": (
                            previous["size_per_dpu_bytes"]
                            if previous is not None
                            and previous["op"] == "dpu_push_xfer"
                            else "0"
                        ),
                        "previous_sdk_mux_domain_class": "COLLECTION",
                        "previous_sdk_subop": (
                            "NONE"
                            if previous is None or previous["subop"] == ""
                            else previous["subop"]
                        ),
                        "previous_sdk_target_space": (
                            "NONE"
                            if previous is None
                            or previous["target_space"] == ""
                            else previous["target_space"]
                        ),
                        "phase_class": phase_class(op, warmup),
                        "source_buffer_reuse_class": source_buffer_reuse_class(
                            use_count
                        ),
                        "target_region_reuse_class": target_region_reuse_class(
                            use_count
                        ),
                        "source_buffer_use_count_before": str(use_count),
                        "target_region_access_count_before": str(use_count),
                        "diagnostic_copy_ordinal": copy_ordinal,
                        "mram_push_ordinal_since_launch": str(mram_ordinal),
                        "size_per_dpu_bytes": str(size),
                        "total_logical_bytes": str(sum(logical)),
                        "total_transfer_bytes": str(size * 64),
                        "target_symbol": str(expectation["target_symbol"]),
                        "offset_bytes": str(expectation["offset"]),
                    }
                )
                row["previous_sdk_op_class"] = previous_sdk_op_class(row)
                row["transport_key"] = transport_key(row)
                for dpu_id in range(64):
                    detail = {field: "" for field in DPU_FIELDS}
                    detail.update(
                        {
                            "run_id": row["run_id"],
                            "repeat_id": row["repeat_id"],
                            "event_id": row["event_id"],
                            "configured_dpus": "64",
                            "actual_ranks": "1",
                            "num_tasklets": "16",
                            "iteration": iteration,
                            "warmup": warmup,
                            "op": op,
                            "subop": subop,
                            "direction": direction,
                            "global_dpu_id": str(dpu_id),
                            "rank_ordinal": "0",
                            "dpu_id_in_rank": str(dpu_id),
                            "sdk_physical_rank_id": "12288",
                            "dpu_sysfs_rank_id": "0",
                            "dpu_rank_numa_node": "0",
                            "dpu_channel_id": "1",
                            "dpu_ci_id": str(dpu_id // 8),
                            "dpu_member_id": str(dpu_id % 8),
                            "logical_bytes": str(logical[dpu_id]),
                            "transfer_bytes": str(size),
                        }
                    )
                    details.append(detail)
            events.append(row)
            previous = row

        with event_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(EVENT_FIELDS))
            writer.writeheader()
            writer.writerows(events)
        with detail_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(DPU_FIELDS))
            writer.writeheader()
            writer.writerows(details)
        return event_path

    def test_accepts_complete_v9_collection_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.create_trace(Path(directory))
            summary = validate(path, expected_sysfs_ranks={0})
        self.assertEqual(summary["transfer_rows"], 16)
        self.assertEqual(summary["dpu_detail_rows"], 1024)

    def test_accepts_vector_then_matrix_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.create_trace(
                Path(directory), "VECTOR_THEN_MATRIX"
            )
            summary = validate(
                path,
                expected_sysfs_ranks={0},
                expected_transfer_order_variant="VECTOR_THEN_MATRIX",
            )
        self.assertEqual(summary["transfer_order_variant"], "VECTOR_THEN_MATRIX")

    def test_accepts_identical_vector_replay_without_key_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.create_trace(
                Path(directory), vector_replay_mode="IDENTICAL_REPLAY"
            )
            summary = validate(
                path,
                expected_sysfs_ranks={0},
                expected_transfer_order_variant="MATRIX_THEN_VECTOR",
                expected_vector_replay_mode="IDENTICAL_REPLAY",
            )
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
        vector_rows = [
            row
            for row in rows
            if row["subop"] == "input_vector" and row["iteration"] == "1"
        ]
        self.assertEqual(summary["event_rows"], 27)
        self.assertEqual(summary["transfer_rows"], 20)
        self.assertEqual(summary["dpu_detail_rows"], 1280)
        self.assertEqual(
            [row["diagnostic_copy_ordinal"] for row in vector_rows],
            ["PRIMARY", "IDENTICAL_REPLAY"],
        )
        self.assertEqual(
            [row["mram_push_ordinal_since_launch"] for row in vector_rows],
            ["2", "3"],
        )
        self.assertTrue(
            all(
                "diagnostic_copy_ordinal" not in row["transport_key"]
                for row in vector_rows
            )
        )
        self.assertTrue(
            all(
                "mram_push_ordinal_since_launch" not in row["transport_key"]
                for row in vector_rows
            )
        )

    def test_rejects_unexpected_rank_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.create_trace(Path(directory))
            with self.assertRaisesRegex(ValueError, "sysfs ranks"):
                validate(path, expected_sysfs_ranks={1})

    def test_rejects_empty_lifecycle_predecessor_subop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.create_trace(Path(directory))
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                fields = list(reader.fieldnames or [])
                rows = list(reader)
            rows[2]["previous_sdk_subop"] = ""
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "predecessor subop"):
                validate(path, expected_sysfs_ranks={0})


if __name__ == "__main__":
    unittest.main()
