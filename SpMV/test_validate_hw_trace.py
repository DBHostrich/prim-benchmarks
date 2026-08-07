#!/usr/bin/env python3

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import validate_hw_trace
from transport_key import (
    call_context,
    derive_hardware_contexts,
    logical_distribution_class,
    offset_feature,
    phase_class,
    physical_dpu_identity,
    same_source_across_group,
    sdk_api_kind,
    transport_key,
)

EVENT_FIELDS = sorted(validate_hw_trace.EVENT_FIELDS)


def distribute_aligned(total: int, count: int) -> list[int]:
    units, remainder = divmod(total // 8, count)
    return [8 * (units + int(i < remainder)) for i in range(count)]


class SpmvTraceValidatorTest(unittest.TestCase):
    def write_trace(
        self, root: Path, nr_dpus: int, tasklets: int
    ) -> Path:
        expected = validate_hw_trace.expected_metrics(nr_dpus)
        nonempty = expected["copy_from"]
        partition_sizes = distribute_aligned(
            expected["h2d_bytes"] - nr_dpus * 24, nonempty * 3
        )
        output_sizes = distribute_aligned(expected["d2h_bytes"], nonempty)
        rows: list[dict[str, str]] = []
        op_counts: dict[str, int] = defaultdict(int)
        dpu_op_counts: dict[tuple[str, int], int] = defaultdict(int)
        timestamp = 1000
        partition_index = 0

        def add(
            op: str,
            subop: str = "",
            direction: str = "",
            dpu_id: int | None = None,
            transfer_bytes: int = 0,
            offset_bytes: int = 0,
        ) -> None:
            nonlocal timestamp
            transfer = dpu_id is not None
            row = {
                "run_id": f"SpMV_{nr_dpus}dpu_{tasklets}tl",
                "repeat_id": "1",
                "event_id": str(len(rows)),
                "configured_dpus": str(nr_dpus),
                "actual_ranks": str(expected["actual_ranks"]),
                "num_tasklets": str(tasklets),
                "op": op,
                "direction": direction,
                "sdk_api_kind": sdk_api_kind(op),
                "logical_distribution_class": logical_distribution_class(
                    op, subop
                ),
                "target_space": "MRAM" if transfer else "",
                "transfer_bytes_per_dpu": (
                    str(transfer_bytes) if transfer else ""
                ),
                "active_dpus": "1" if transfer else "",
                "active_ranks": "1" if transfer else "",
                "active_dpus_per_rank": "1" if transfer else "",
                "rank_ordinal": str(dpu_id // 64) if transfer else "",
                "dpu_id_in_rank": str(dpu_id % 64) if transfer else "",
                "sdk_physical_rank_id": (
                    str(12288 + dpu_id // 64) if transfer else ""
                ),
                "dpu_sysfs_rank_id": str(dpu_id // 64) if transfer else "",
                "dpu_rank_numa_node": "0" if transfer else "",
                "dpu_channel_id": (
                    str(dpu_id // 256 + 1) if transfer else ""
                ),
                "sdk_slice_id": str((dpu_id % 64) // 8) if transfer else "",
                "sdk_member_id": str(dpu_id % 8) if transfer else "",
                "dpu_ci_id": str((dpu_id % 64) // 8) if transfer else "",
                "dpu_member_id": str(dpu_id % 8) if transfer else "",
                "physical_dpu_identity": "",
                "cpu_dpu_numa_relation": "LOCAL" if transfer else "",
                "same_source_across_group": same_source_across_group(
                    op, subop
                ),
                "phase_class": phase_class(op, subop),
                "subop": subop,
                "global_dpu_id": str(dpu_id) if transfer else "",
                "target_symbol": (
                    "DPU_MRAM_HEAP_POINTER_NAME" if transfer else ""
                ),
                "offset_bytes": str(offset_bytes) if transfer else "",
                "offset_feature": "",
                "logical_bytes": str(transfer_bytes) if transfer else "",
                "transfer_bytes": str(transfer_bytes) if transfer else "",
                "host_buffer_address": (
                    str(0x100000 + offset_bytes + (0 if subop == "input_vector" else (dpu_id or 0) * 4096))
                    if transfer else ""
                ),
                "host_buffer_page_offset": "",
                "host_buffer_reuse_class": "",
                "previous_sdk_op": "",
                "previous_sdk_direction": "",
                "previous_sdk_transfer_bytes": "",
                "previous_sdk_topology_relation": "",
                "ns_since_previous_sdk_event": "",
                "previous_dpu_direction": "",
                "previous_dpu_transfer_bytes": "",
                "previous_dpu_target_relation": "",
                "launches_since_previous_dpu_transfer": "",
                "target_region_reuse_class": "",
                "op_call_index": str(op_counts[op]),
                "dpu_op_call_index": "",
                "process_state": "fresh_process",
                "pretrace_warmup_runs": "5",
                "host_numa_node": "0",
                "call_context": "",
                "transport_key": "",
                "host_start_ns": str(timestamp),
                "host_end_ns": str(timestamp + 100),
                "measured_ns": "100",
            }
            op_counts[op] += 1
            if transfer:
                key = (op, dpu_id)
                row["dpu_op_call_index"] = str(dpu_op_counts[key])
                dpu_op_counts[key] += 1
            row["offset_feature"] = offset_feature(row)
            row["call_context"] = call_context(row)
            rows.append(row)
            timestamp += 200

        add("dpu_alloc")
        add("dpu_load")
        for dpu_id in range(nr_dpus):
            if dpu_id < nonempty:
                for subop, offset in (
                    ("row_ptrs", 24),
                    ("nonzeros", 4096),
                    ("input_vector", 8192),
                ):
                    size = partition_sizes[partition_index]
                    partition_index += 1
                    add("dpu_copy_to", subop, "TO_DPU", dpu_id, size, offset)
            add("dpu_copy_to", "params", "TO_DPU", dpu_id, 24, 0)
        add("dpu_launch", "sync")
        for dpu_id, size in enumerate(output_sizes):
            add(
                "dpu_copy_from",
                "output_vector",
                "FROM_DPU",
                dpu_id,
                size,
                12288,
            )
        add("dpu_free")

        for row, context in zip(rows, derive_hardware_contexts(rows)):
            row.update(context)
            row["physical_dpu_identity"] = physical_dpu_identity(row)
            row["transport_key"] = transport_key(row)

        path = root / "trace_01.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=EVENT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_valid_256_dpu_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_trace(Path(directory), 256, 1)
            summary = validate_hw_trace.validate(path)
            self.assertEqual(summary["events"], 1276)
            self.assertEqual(summary["h2d_bytes"], 37_800_320)
            self.assertEqual(summary["d2h_bytes"], 115_696)

    def test_valid_512_dpu_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_trace(Path(directory), 512, 16)
            summary = validate_hw_trace.validate(path)
            self.assertEqual(summary["events"], 2512)
            self.assertEqual(summary["h2d_bytes"], 66_153_944)
            self.assertEqual(summary["d2h_bytes"], 115_696)

    def test_input_vector_uses_shared_replication_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_trace(Path(directory), 256, 1)
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            row = next(row for row in rows if row["subop"] == "input_vector")
            self.assertEqual(
                row["logical_distribution_class"], "SHARED_REPLICATION"
            )
            self.assertEqual(row["same_source_across_group"], "1")
            self.assertEqual(row["phase_class"], "INIT")
            self.assertEqual(row["active_dpus"], "1")

    def test_output_vector_uses_partitioned_finalize_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_trace(Path(directory), 256, 1)
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            row = next(row for row in rows if row["subop"] == "output_vector")
            self.assertEqual(
                row["logical_distribution_class"], "PARTITIONED_GATHER"
            )
            self.assertEqual(row["phase_class"], "FINALIZE")

    def test_rejects_transport_key_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_trace(Path(directory), 256, 1)
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            transfer = next(row for row in rows if row["op"] == "dpu_copy_to")
            transfer["transport_key"] = transfer["transport_key"].replace(
                "active_dpus=1", "active_dpus=2"
            )
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=EVENT_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "invalid transport_key"):
                validate_hw_trace.validate(path)


if __name__ == "__main__":
    unittest.main()
