from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from analyze_vector_replay_delay_probe import analyze


SCHEDULES = (
    (0, 0, 0, 0),
    (0, 0, 1000, 1000),
    (0, 1000, 0, 1000),
    (0, 1000, 1000, 0),
    (1000, 0, 0, 1000),
    (1000, 0, 1000, 0),
    (1000, 1000, 0, 0),
    (1000, 1000, 1000, 1000),
)


class AnalyzeVectorReplayDelayProbeTests(unittest.TestCase):
    def write_trace(self, root: Path, repeat: int) -> None:
        directory = root / "GEMV_128dpu_16tl_VECTOR_REPLAY"
        directory.mkdir(parents=True, exist_ok=True)
        schedule = SCHEDULES[(repeat - 1) % len(SCHEDULES)]
        slow_primary = repeat <= 8
        rows: list[dict[str, str]] = []
        event_id = 0
        clock_ns = 1_000_000
        for iteration, delay_us in enumerate(schedule):
            context = "FIRST_USE" if iteration == 0 else "REUSED"
            rows.append(
                {
                    "run_id": directory.name,
                    "repeat_id": str(repeat),
                    "event_id": str(event_id),
                    "op": "dpu_push_xfer",
                    "subop": "input_matrix",
                    "measured_ns": "100",
                    "vector_replay_mode": "IDENTICAL_REPLAY",
                }
            )
            event_id += 1
            primary_ns = 200 if slow_primary else 100
            primary_start = clock_ns
            primary_end = primary_start + primary_ns
            rows.append(
                {
                    "run_id": directory.name,
                    "repeat_id": str(repeat),
                    "event_id": str(event_id),
                    "op": "dpu_push_xfer",
                    "subop": "input_vector",
                    "direction": "TO_DPU",
                    "target_space": "MRAM",
                    "target_symbol": "DPU_MRAM_HEAP_POINTER_NAME",
                    "offset_bytes": "32768",
                    "transfer_bytes_per_dpu": "32768",
                    "total_logical_bytes": "4194304",
                    "total_transfer_bytes": "4194304",
                    "same_source_across_group": "1",
                    "allocated_topology_signature": "r0@n0@c1|r4@n0@c2",
                    "iteration": str(iteration),
                    "source_buffer_reuse_class": context,
                    "source_buffer_use_count_before": str(2 * iteration),
                    "diagnostic_copy_ordinal": "PRIMARY",
                    "mram_push_ordinal_since_launch": "2",
                    "replay_delay_requested_us": "0",
                    "previous_sdk_subop": "input_matrix",
                    "transport_key": f"key_{context}",
                    "measured_ns": str(primary_ns),
                    "wall_minus_thread_cpu_ns": str(primary_ns - 10),
                    "cpu_id_start": "0",
                    "cpu_id_end": "0",
                    "involuntary_context_switch_delta": "0",
                    "host_start_ns": str(primary_start),
                    "host_end_ns": str(primary_end),
                    "vector_replay_mode": "IDENTICAL_REPLAY",
                }
            )
            event_id += 1
            replay_ns = 200 if slow_primary and delay_us == 0 else 90
            replay_start = primary_end + delay_us * 1000 + 10_000
            rows.append(
                {
                    **rows[-1],
                    "event_id": str(event_id),
                    "diagnostic_copy_ordinal": "IDENTICAL_REPLAY",
                    "mram_push_ordinal_since_launch": "3",
                    "replay_delay_requested_us": str(delay_us),
                    "previous_sdk_subop": "input_vector",
                    "source_buffer_use_count_before": str(2 * iteration + 1),
                    "measured_ns": str(replay_ns),
                    "wall_minus_thread_cpu_ns": str(replay_ns - 10),
                    "host_start_ns": str(replay_start),
                    "host_end_ns": str(replay_start + replay_ns),
                }
            )
            event_id += 1
            clock_ns = replay_start + replay_ns + 1_000_000

        trace_path = directory / f"trace_{repeat:02d}.csv"
        fields = sorted({field for row in rows for field in row})
        with trace_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with (directory / f"heartbeat_{repeat:02d}.csv").open(
            "w", newline=""
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["planned_raw_ns", "actual_raw_ns", "lateness_ns"],
            )
            writer.writeheader()

    def test_detects_time_decaying_transfer_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for repeat in range(1, 25):
                self.write_trace(root, repeat)
            outputs = analyze(root)
        self.assertEqual(len(outputs["vector_replay_delay_events"]), 96)
        self.assertEqual(len(outputs["vector_replay_delay_summary"]), 4)
        self.assertEqual(len(outputs["vector_replay_delay_iteration_summary"]), 8)
        self.assertEqual(
            {row["decision"] for row in outputs["vector_replay_delay_comparison"]},
            {"SUPPORTS_TIME_DECAYING_TRANSFER_EPISODE"},
        )


if __name__ == "__main__":
    unittest.main()
