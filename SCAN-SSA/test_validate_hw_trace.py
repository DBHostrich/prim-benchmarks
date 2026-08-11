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
    expected_layout,
    expected_sequence,
    expected_transfer,
    validate,
)


class ValidateHardwareTraceTests(unittest.TestCase):
    def test_c_trace_uses_canonical_mram_target_label(self) -> None:
        source = (Path(__file__).with_name("host") / "app.c").read_text()
        self.assertEqual(
            source.count('"MRAM", "DPU_MRAM_HEAP_POINTER_NAME"'), 2
        )

    def create_trace(self, root: Path) -> Path:
        event_path = root / "trace_01.csv"
        detail_path = root / "trace_01_dpus.csv"
        events = []
        details = []
        nr_dpus = 64
        raw, rounded, data_logical = expected_layout(nr_dpus)
        previous = None

        for event_id, semantic in enumerate(expected_sequence()):
            op, subop, direction, iteration, warmup = semantic
            row = {field: "" for field in EVENT_FIELDS}
            row.update(
                {
                    "run_id": "SCAN_64dpu_16tl",
                    "repeat_id": "1",
                    "event_id": str(event_id),
                    "configured_dpus": "64",
                    "actual_ranks": "1",
                    "num_tasklets": "16",
                    "input_size": "251658240",
                    "input_size_per_dpu": str(raw),
                    "input_size_per_dpu_round": str(rounded),
                    "scaling_mode": "STRONG",
                    "op": op,
                    "subop": subop,
                    "direction": direction,
                    "iteration": iteration,
                    "warmup": warmup,
                    "host_numa_node": "0",
                    "process_state": "interleaved_fresh_process",
                    "host_binding_mode": "NODE_ONLY",
                    "host_cpu_list": "numa_node_0",
                    "algorithm_variant": "SCAN_SSA",
                    "scaling_mode_label": "STRONG",
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
                expected = expected_transfer(
                    subop, nr_dpus, rounded, data_logical
                )
                size = int(expected["size"])
                logical = list(expected["logical"])
                row.update(
                    {
                        "sdk_api_kind": sdk_api_kind(op),
                        "timing_scope": "PUSH_ONLY",
                        "logical_distribution_class":
                            logical_distribution_class(op, subop),
                        "target_space": str(expected["target_space"]),
                        "transfer_bytes_per_dpu": str(size),
                        "active_dpus": "64",
                        "active_ranks": "1",
                        "active_dpus_per_rank": "64",
                        "rank_ordinal": "ALL",
                        "dpu_id_in_rank": "ALL",
                        "same_source_across_group":
                            same_source_across_group(op, subop),
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
                        "previous_sdk_op": previous["op"],
                        "previous_sdk_direction":
                            previous["direction"] or "NONE",
                        "previous_sdk_transfer_bytes":
                            previous["size_per_dpu_bytes"]
                            if previous["op"] == "dpu_push_xfer" else "0",
                        "previous_sdk_mux_domain_class": "COLLECTION",
                        "previous_sdk_subop": previous["subop"] or "NONE",
                        "previous_sdk_target_space":
                            previous["target_space"] or "NONE",
                        "source_buffer_reuse_class":
                            source_buffer_reuse_class(iteration_number),
                        "target_region_reuse_class":
                            target_region_reuse_class(iteration_number),
                        "source_buffer_use_count_before":
                            str(iteration_number),
                        "target_region_access_count_before":
                            str(iteration_number),
                        "phase_class": phase_class(op, warmup),
                        "diagnostic_copy_ordinal": "NONE",
                        "mram_push_ordinal_since_launch":
                            "1" if subop in {"input_data", "output_data"}
                            else "0",
                        "replay_delay_requested_us": "0",
                        "size_per_dpu_bytes": str(size),
                        "total_logical_bytes": str(sum(logical)),
                        "total_transfer_bytes": str(size * nr_dpus),
                        "target_symbol": str(expected["target_symbol"]),
                        "offset_bytes": str(expected["offset"]),
                    }
                )
                row["previous_sdk_op_class"] = previous_sdk_op_class(row)
                row["transport_key"] = transport_key(row)
                for dpu_id in range(nr_dpus):
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
            summary = validate(
                self.create_trace(Path(directory)), expected_sysfs_ranks={0}
            )
        self.assertEqual(summary["event_rows"], 31)
        self.assertEqual(summary["transfer_rows"], 20)
        self.assertEqual(summary["dpu_detail_rows"], 1280)

    def test_rejects_faulty_rank_5(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.create_trace(Path(directory))
            detail_path = path.with_name("trace_01_dpus.csv")
            with detail_path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                row["dpu_sysfs_rank_id"] = "5"
            with detail_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "rank 5"):
                validate(path)


if __name__ == "__main__":
    unittest.main()

