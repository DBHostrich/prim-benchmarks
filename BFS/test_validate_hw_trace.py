#!/usr/bin/env python3

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import validate_hw_trace


FIELDNAMES = [
    "run_id",
    "repeat_id",
    "event_id",
    "configured_dpus",
    "actual_ranks",
    "num_tasklets",
    "op",
    "subop",
    "bfs_level",
    "direction",
    "global_dpu_id",
    "rank_ordinal",
    "dpu_id_in_rank",
    "target_space",
    "target_symbol",
    "offset_bytes",
    "logical_bytes",
    "transfer_bytes",
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
            rows.append(
                {
                    "run_id": f"BFS_{nr_dpus}dpu_1tl",
                    "repeat_id": "1",
                    "event_id": str(event_id),
                    "configured_dpus": str(nr_dpus),
                    "actual_ranks": str(actual_ranks),
                    "num_tasklets": "1",
                    "op": op,
                    "subop": subop,
                    "bfs_level": bfs_level,
                    "direction": direction,
                    "global_dpu_id": str(dpu_id) if has_dpu else "",
                    "rank_ordinal": str(dpu_id // 64) if has_dpu else "",
                    "dpu_id_in_rank": str(dpu_id % 64) if has_dpu else "",
                    "target_space": "MRAM" if has_dpu else "",
                    "target_symbol": (
                        "DPU_MRAM_HEAP_POINTER_NAME" if has_dpu else ""
                    ),
                    "offset_bytes": "0" if has_dpu else "",
                    "logical_bytes": (
                        str(logical_bytes) if logical_bytes is not None else ""
                    ),
                    "transfer_bytes": (
                        str(transfer_bytes) if transfer_bytes is not None else ""
                    ),
                    "host_start_ns": str(timestamp),
                    "host_end_ns": str(timestamp + 100),
                    "measured_ns": "100",
                }
            )
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


if __name__ == "__main__":
    unittest.main()
