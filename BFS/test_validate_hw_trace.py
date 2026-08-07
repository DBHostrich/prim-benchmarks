#!/usr/bin/env python3

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
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
    transport_key_v7_full,
    transport_key_v6_full,
)


FIELDNAMES = [
    "run_id",
    "repeat_id",
    "event_id",
    "configured_dpus",
    "actual_ranks",
    "num_tasklets",
    "op",
    "direction",
    "sdk_api_kind",
    "logical_distribution_class",
    "target_space",
    "transfer_bytes_per_dpu",
    "active_dpus",
    "active_ranks",
    "active_dpus_per_rank",
    "rank_ordinal",
    "dpu_id_in_rank",
    "sdk_physical_rank_id",
    "dpu_sysfs_rank_id",
    "dpu_rank_numa_node",
    "dpu_channel_id",
    "sdk_slice_id",
    "sdk_member_id",
    "dpu_ci_id",
    "dpu_member_id",
    "physical_dpu_identity",
    "cpu_dpu_numa_relation",
    "same_source_across_group",
    "phase_class",
    "subop",
    "bfs_level",
    "global_dpu_id",
    "target_symbol",
    "offset_bytes",
    "offset_feature",
    "logical_bytes",
    "transfer_bytes",
    "host_buffer_address",
    "host_buffer_page_offset",
    "host_buffer_reuse_class",
    "previous_sdk_op",
    "previous_sdk_direction",
    "previous_sdk_transfer_bytes",
    "previous_sdk_topology_relation",
    "ns_since_previous_sdk_event",
    "previous_dpu_direction",
    "previous_dpu_transfer_bytes",
    "previous_dpu_target_relation",
    "launches_since_previous_dpu_transfer",
    "target_region_reuse_class",
    "op_call_index",
    "dpu_op_call_index",
    "process_state",
    "pretrace_warmup_runs",
    "host_numa_node",
    "call_context",
    "transport_key",
    "host_start_ns",
    "host_end_ns",
    "measured_ns",
]


def round_up_to_8(value: int) -> int:
    return ((value + 7) // 8) * 8


class BfsTraceValidatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        graph_path = Path(__file__).parent / "data" / "loc-gowalla_edges.txt"
        with graph_path.open() as stream:
            num_nodes, num_columns, num_edges = map(int, stream.readline().split())
            cls.num_nodes = ((max(num_nodes, num_columns) + 63) // 64) * 64
            cls.degrees = [0] * cls.num_nodes
            observed_edges = 0
            for line in stream:
                node, _ = map(int, line.split())
                cls.degrees[node] += 1
                observed_edges += 1
        if observed_edges != num_edges:
            raise AssertionError(
                f"graph edges={observed_edges}, header reports {num_edges}"
            )

    def write_trace(self, path: Path, nr_dpus: int) -> None:
        rows = []
        timestamp = 1_000_000
        event_id = 0
        actual_ranks = nr_dpus // 64
        op_call_counts: dict[str, int] = {}
        dpu_op_call_counts: dict[tuple[str, int], int] = {}

        def append(
            op: str,
            subop: str = "",
            bfs_level: str = "",
            direction: str = "",
            dpu_id: int | None = None,
            logical_bytes: int | None = None,
        ) -> None:
            nonlocal event_id, timestamp
            has_dpu = dpu_id is not None
            transfer_bytes = (
                round_up_to_8(logical_bytes)
                if logical_bytes is not None
                else None
            )
            op_call_index = op_call_counts.get(op, 0)
            op_call_counts[op] = op_call_index + 1
            dpu_op_call_index = None
            if has_dpu and op in {"dpu_copy_to", "dpu_copy_from"}:
                key = (op, dpu_id)
                dpu_op_call_index = dpu_op_call_counts.get(key, 0)
                dpu_op_call_counts[key] = dpu_op_call_index + 1
            row = {
                    "run_id": f"BFS_{nr_dpus}dpu_1tl",
                    "repeat_id": "1",
                    "event_id": str(event_id),
                    "configured_dpus": str(nr_dpus),
                    "actual_ranks": str(actual_ranks),
                    "num_tasklets": "1",
                    "op": op,
                    "direction": direction,
                    "sdk_api_kind": sdk_api_kind(op),
                    "logical_distribution_class": logical_distribution_class(
                        op, subop
                    ),
                    "target_space": "MRAM" if has_dpu else "",
                    "transfer_bytes_per_dpu": (
                        str(transfer_bytes) if has_dpu else ""
                    ),
                    "active_dpus": "1" if has_dpu else "",
                    "active_ranks": "1" if has_dpu else "",
                    "active_dpus_per_rank": "1" if has_dpu else "",
                    "rank_ordinal": str(dpu_id // 64) if has_dpu else "",
                    "dpu_id_in_rank": str(dpu_id % 64) if has_dpu else "",
                    "sdk_physical_rank_id": (
                        str(12_288 + dpu_id // 64) if has_dpu else ""
                    ),
                    "dpu_sysfs_rank_id": (
                        str(dpu_id // 64) if has_dpu else ""
                    ),
                    "dpu_rank_numa_node": "0" if has_dpu else "",
                    "dpu_channel_id": (
                        str(1 + (dpu_id // 64) // 4) if has_dpu else ""
                    ),
                    "sdk_slice_id": str((dpu_id % 64) // 8) if has_dpu else "",
                    "sdk_member_id": str(dpu_id % 8) if has_dpu else "",
                    "dpu_ci_id": str((dpu_id % 64) // 8) if has_dpu else "",
                    "dpu_member_id": str(dpu_id % 8) if has_dpu else "",
                    "physical_dpu_identity": "",
                    "cpu_dpu_numa_relation": "LOCAL" if has_dpu else "",
                    "same_source_across_group": same_source_across_group(
                        op, subop
                    ),
                    "phase_class": phase_class(op, subop),
                    "subop": subop,
                    "bfs_level": bfs_level,
                    "global_dpu_id": str(dpu_id) if has_dpu else "",
                    "target_symbol": (
                        "DPU_MRAM_HEAP_POINTER_NAME" if has_dpu else ""
                    ),
                    "offset_bytes": (
                        str(
                            {
                                "params_init": 0,
                                "params_level": 0,
                                "node_ptrs": 1_024,
                                "neighbor_idxs": 8_192,
                                "node_level_init": 16_384,
                                "node_level_result": 16_384,
                                "visited_init": 32_768,
                                "frontier_init": 65_536,
                                "frontier_broadcast": 65_536,
                                "frontier_result": 65_536,
                            }[subop]
                        )
                        if has_dpu else ""
                    ),
                    "offset_feature": "",
                    "logical_bytes": (
                        str(logical_bytes) if logical_bytes is not None else ""
                    ),
                    "transfer_bytes": (
                        str(transfer_bytes) if transfer_bytes is not None else ""
                    ),
                    "host_buffer_address": (
                        str(
                            {
                                "visited_init": 0x100000,
                                "frontier_init": 0x200000,
                                "frontier_broadcast": 0x300000,
                                "frontier_result": (
                                    0x300000 if dpu_id == 0 else 0x200000
                                ),
                            }.get(subop, 0x400000 + event_id * 64)
                        )
                        if has_dpu else ""
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
                    "op_call_index": str(op_call_index),
                    "dpu_op_call_index": (
                        str(dpu_op_call_index)
                        if dpu_op_call_index is not None
                        else ""
                    ),
                    "process_state": "fresh_process",
                    "pretrace_warmup_runs": "5",
                    "host_numa_node": "0",
                    "call_context": "",
                    "transport_key": "",
                    "host_start_ns": str(timestamp),
                    "host_end_ns": str(timestamp + 100),
                    "measured_ns": "100",
                }
            row["offset_feature"] = offset_feature(row)
            row["call_context"] = call_context(row)
            row["physical_dpu_identity"] = physical_dpu_identity(row)
            rows.append(row)
            timestamp += 200
            event_id += 1

        append("dpu_alloc")
        append("dpu_load")

        nodes_per_dpu = self.num_nodes // nr_dpus
        frontier_bytes = self.num_nodes // 64 * 8
        for dpu_id in range(nr_dpus):
            start_node = dpu_id * nodes_per_dpu
            dpu_edges = sum(
                self.degrees[start_node : start_node + nodes_per_dpu]
            )
            append(
                "dpu_copy_to",
                "node_ptrs",
                direction="TO_DPU",
                dpu_id=dpu_id,
                logical_bytes=(nodes_per_dpu + 1) * 4,
            )
            append(
                "dpu_copy_to",
                "neighbor_idxs",
                direction="TO_DPU",
                dpu_id=dpu_id,
                logical_bytes=dpu_edges * 4,
            )
            append(
                "dpu_copy_to",
                "node_level_init",
                direction="TO_DPU",
                dpu_id=dpu_id,
                logical_bytes=nodes_per_dpu * 4,
            )
            append(
                "dpu_copy_to",
                "visited_init",
                direction="TO_DPU",
                dpu_id=dpu_id,
                logical_bytes=frontier_bytes,
            )
            append(
                "dpu_copy_to",
                "frontier_init",
                direction="TO_DPU",
                dpu_id=dpu_id,
                logical_bytes=frontier_bytes,
            )
            append(
                "dpu_copy_to",
                "params_init",
                direction="TO_DPU",
                dpu_id=dpu_id,
                logical_bytes=44,
            )

        for level in range(1, validate_hw_trace.EXPECTED_LEVELS + 1):
            append("dpu_launch", "bfs_level", str(level))
            for dpu_id in range(nr_dpus):
                append(
                    "dpu_copy_from",
                    "frontier_result",
                    str(level),
                    "FROM_DPU",
                    dpu_id,
                    frontier_bytes,
                )
            if level < validate_hw_trace.EXPECTED_LEVELS:
                next_level = str(level + 1)
                for dpu_id in range(nr_dpus):
                    append(
                        "dpu_copy_to",
                        "frontier_broadcast",
                        next_level,
                        "TO_DPU",
                        dpu_id,
                        frontier_bytes,
                    )
                    append(
                        "dpu_copy_to",
                        "params_level",
                        next_level,
                        "TO_DPU",
                        dpu_id,
                        44,
                    )

        for dpu_id in range(nr_dpus):
            append(
                "dpu_copy_from",
                "node_level_result",
                direction="FROM_DPU",
                dpu_id=dpu_id,
                logical_bytes=nodes_per_dpu * 4,
            )
        append("dpu_free")

        for row, context in zip(rows, derive_hardware_contexts(rows)):
            row.update(context)
            row["transport_key"] = transport_key(row)

        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

    def test_valid_256_dpu_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            self.write_trace(path, 256)
            summary = validate_hw_trace.validate(path)
            self.assertEqual(summary["events"], 8_973)
            self.assertEqual(summary["h2d_transfer_bytes"], 78_506_896)
            self.assertEqual(summary["d2h_transfer_bytes"], 63_700_992)

    def test_valid_64_dpu_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            self.write_trace(path, 64)
            summary = validate_hw_trace.validate(path)
            self.assertEqual(summary["actual_ranks"], 1)
            self.assertEqual(summary["events"], 2_253)

    def test_valid_128_dpu_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            self.write_trace(path, 128)
            summary = validate_hw_trace.validate(path)
            self.assertEqual(summary["actual_ranks"], 2)
            self.assertEqual(summary["events"], 4_493)

    def test_rejects_partial_rank_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "whole number"):
            validate_hw_trace.expected_metrics(96)

    def test_valid_512_dpu_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            self.write_trace(path, 512)
            summary = validate_hw_trace.validate(path)
            self.assertEqual(summary["events"], 17_933)
            self.assertEqual(summary["h2d_transfer_bytes"], 147_838_376)
            self.assertEqual(summary["d2h_transfer_bytes"], 126_615_552)

    def test_rejects_wrong_level(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            self.write_trace(path, 256)
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            launch = next(row for row in rows if row["op"] == "dpu_launch")
            launch["bfs_level"] = "7"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "semantic tuple"):
                validate_hw_trace.validate(path)

    def test_rejects_wrong_transport_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            self.write_trace(path, 256)
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            copy_row = next(row for row in rows if row["op"] == "dpu_copy_to")
            copy_row["transport_key"] += ";corrupt=1"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "invalid transport_key"):
                validate_hw_trace.validate(path)

    def test_legacy_v6_key_requires_explicit_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            self.write_trace(path, 64)
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                if row["op"] in {"dpu_copy_to", "dpu_copy_from"}:
                    row["transport_key"] = transport_key_v6_full(row)
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(ValueError, "invalid transport_key"):
                validate_hw_trace.validate(path)
            summary = validate_hw_trace.validate(
                path, allow_legacy_transport_key=True
            )
            self.assertEqual(summary["configured_dpus"], 64)

    def test_distribution_and_same_source_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            self.write_trace(path, 256)
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))

            by_subop = {}
            for row in rows:
                by_subop.setdefault(row["subop"], row)
            self.assertEqual(
                by_subop["frontier_broadcast"]["logical_distribution_class"],
                "SHARED_REPLICATION",
            )
            self.assertEqual(
                by_subop["frontier_broadcast"]["same_source_across_group"],
                "1",
            )
            self.assertEqual(
                by_subop["neighbor_idxs"]["logical_distribution_class"],
                "PARTITIONED_SCATTER",
            )
            self.assertEqual(
                by_subop["neighbor_idxs"]["same_source_across_group"], "0"
            )
            self.assertEqual(
                by_subop["frontier_result"]["logical_distribution_class"],
                "REDUCTION_GATHER",
            )
            self.assertEqual(
                by_subop["node_level_result"]["logical_distribution_class"],
                "PARTITIONED_GATHER",
            )
            self.assertEqual(by_subop["visited_init"]["phase_class"], "INIT")
            self.assertEqual(by_subop["frontier_init"]["phase_class"], "INIT")
            self.assertEqual(
                by_subop["frontier_broadcast"]["phase_class"], "ITERATIVE"
            )
            self.assertEqual(
                by_subop["node_level_result"]["phase_class"], "FINALIZE"
            )

    def test_transport_key_has_exact_ordered_hardware_context_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            self.write_trace(path, 256)
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            row = next(
                item
                for item in rows
                if item["subop"] == "frontier_broadcast"
                and item["global_dpu_id"] == "17"
            )
            self.assertEqual(row["sdk_api_kind"], "SINGLE_COPY")
            self.assertEqual(row["transfer_bytes_per_dpu"], "24576")
            self.assertEqual(row["active_dpus"], "1")
            self.assertEqual(row["active_ranks"], "1")
            self.assertEqual(row["active_dpus_per_rank"], "1")
            self.assertEqual(row["phase_class"], "ITERATIVE")
            self.assertEqual(row["sdk_physical_rank_id"], "12288")
            self.assertEqual(row["dpu_sysfs_rank_id"], "0")
            self.assertEqual(row["dpu_rank_numa_node"], "0")
            self.assertEqual(row["dpu_channel_id"], "1")
            self.assertEqual(row["sdk_slice_id"], "2")
            self.assertEqual(row["sdk_member_id"], "1")
            self.assertEqual(row["dpu_ci_id"], "2")
            self.assertEqual(row["dpu_member_id"], "1")
            self.assertEqual(row["cpu_dpu_numa_relation"], "LOCAL")
            self.assertEqual(
                row["physical_dpu_identity"],
                "numa:0/channel:1/rank:0/ci:2/member:1",
            )
            self.assertEqual(row["previous_dpu_direction"], "FROM_DPU")
            self.assertEqual(
                row["previous_sdk_topology_relation"], "SAME_MUX_PAIR"
            )
            self.assertEqual(row["previous_dpu_target_relation"], "SAME_REGION")
            self.assertEqual(row["target_region_reuse_class"], "REUSED_REGION")
            self.assertEqual(
                row["transport_key"],
                "v8;op=dpu_copy_to;direction=TO_DPU;"
                "sdk_api_kind=SINGLE_COPY;"
                "logical_distribution_class=SHARED_REPLICATION;"
                "target_space=MRAM;transfer_bytes_per_dpu=24576;"
                "active_dpus=1;active_ranks=1;active_dpus_per_rank=1;"
                "same_source_across_group=1;host_numa_node=0;"
                "dpu_rank_numa_node=0;cpu_dpu_numa_relation=LOCAL;"
                "dpu_channel_id=1;dpu_sysfs_rank_id=0;"
                "dpu_ci_id=2;dpu_member_id=1;allocated_dpus=256;"
                "allocated_ranks=4;"
                "previous_sdk_mux_domain_class=SAME_MUX_DOMAIN",
            )
            self.assertNotIn("rank_ordinal=", row["transport_key"])
            self.assertNotIn("dpu_id_in_rank=", row["transport_key"])
            self.assertNotIn("phase_class=", row["transport_key"])

    def test_sdk_topology_relation_distinguishes_mux_pair_layers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            self.write_trace(path, 256)
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
        frontier_rows = {
            int(row["global_dpu_id"]): row
            for row in rows
            if row["subop"] == "frontier_broadcast" and row["bfs_level"] == "2"
        }
        self.assertEqual(
            frontier_rows[16]["previous_sdk_topology_relation"], "SAME_RANK"
        )
        self.assertEqual(
            frontier_rows[17]["previous_sdk_topology_relation"],
            "SAME_MUX_PAIR",
        )
        self.assertEqual(
            frontier_rows[18]["previous_sdk_topology_relation"], "SAME_SLICE"
        )


if __name__ == "__main__":
    unittest.main()
