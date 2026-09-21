import copy
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_va_transfer_breakdown as validator


class ValidateVaTransferBreakdownTest(unittest.TestCase):
    def make_app_rows(self, run_id="formal_01", repeat=1, dpus=64):
        rows = []
        wall = 1_000
        thread_cpu = 2_000
        process_cpu = 3_000
        for index, component in enumerate(validator.COMPONENTS):
            if component == "PREPARE_C":
                wall = 2_000
                thread_cpu = 2_500
                process_cpu = 3_700
            direction, target, operation = validator.COMPONENT_META[component]
            payload = validator.aligned_per_dpu_elements(2_621_440, dpus) * 4
            per_dpu = 12 if component.endswith("ARGS") else payload
            rows.append({
                "schema_version": validator.APP_SCHEMA,
                "run_id": run_id,
                "repeat_id": str(repeat),
                "configuration": "rank",
                "actual_dpus": str(dpus),
                "actual_ranks": "1",
                "tasklets": "16",
                "block_size_log2": "10",
                "input_elements": "2621440",
                "component": component,
                "direction": direction,
                "target_space": target,
                "operation": operation,
                "wall_start_ns": str(wall),
                "wall_end_ns": str(wall + 100),
                "wall_duration_ns": "100",
                "thread_cpu_start_ns": str(thread_cpu),
                "thread_cpu_end_ns": str(thread_cpu + 50),
                "thread_cpu_duration_ns": "50",
                "process_cpu_start_ns": str(process_cpu),
                "process_cpu_end_ns": str(process_cpu + 70),
                "process_cpu_duration_ns": "70",
                "cpu_start": "4",
                "cpu_end": "4",
                "thread_nvcsw_delta": "0",
                "thread_nivcsw_delta": "0",
                "process_nvcsw_delta": "0",
                "process_nivcsw_delta": "0",
                "bytes_per_dpu": str(per_dpu),
                "aggregate_bytes": str(per_dpu * dpus),
                "result_ok": "1",
                "source_sha256": "source",
                "host_binary_sha256": "host",
                "dpu_binary_sha256": "dpu",
                "_source_path": "app.csv",
            })
            wall += 100
            thread_cpu += 50
            process_cpu += 70
        return rows

    def make_sdk_row(self, sequence, run_id, kind, direction, memory, start, end,
                     per_dpu, dpus=64):
        return {
            "schema_version": validator.SDK_SCHEMA,
            "run_id": run_id,
            "sequence": str(sequence),
            "pid": "100",
            "tid": "101",
            "event_kind": kind,
            "direction": direction,
            "memory_type": memory,
            "bytes_per_dpu": str(per_dpu),
            "aggregate_bytes": str(per_dpu * dpus),
            "rank_id": "0",
            "rank_count": "1",
            "dpu_count": str(dpus),
            "wall_start_ns": str(start),
            "wall_end_ns": str(end),
            "wall_duration_ns": str(end - start),
            "thread_cpu_start_ns": "1000",
            "thread_cpu_end_ns": "1010",
            "thread_cpu_duration_ns": "10",
            "process_cpu_start_ns": "2000",
            "process_cpu_end_ns": "2020",
            "process_cpu_duration_ns": "20",
            "cpu_start": "4",
            "cpu_end": "5",
            "thread_nvcsw_delta": "0",
            "thread_nivcsw_delta": "0",
            "process_nvcsw_delta": "0",
            "process_nivcsw_delta": "0",
            "xfer_flags": "0",
            "return_status": "0",
            "diagnostic_flags": "0",
            "overflow_count": "0",
            "runtime_version": "2025.1",
            "trace_clock": "CLOCK_MONOTONIC_RAW",
            "_source_path": "sdk.csv",
        }

    def make_sdk_rows(self, app_rows):
        indexed = {row["component"]: row for row in app_rows}
        rows = []
        sequence = 0
        for component, direction, memory in (
            ("PUSH_ARGS", "H2D", "WRAM"),
            ("PUSH_A", "H2D", "MRAM"),
            ("PUSH_B", "H2D", "MRAM"),
            ("PUSH_C", "D2H", "MRAM"),
        ):
            app = indexed[component]
            start = int(app["wall_start_ns"]) + 2
            end = int(app["wall_end_ns"]) - 2
            per_dpu = int(app["bytes_per_dpu"])
            rows.append(self.make_sdk_row(
                sequence, app["run_id"], "PUSH_CALL", direction, memory,
                start, end, per_dpu,
            ))
            sequence += 1
            if memory == "MRAM":
                rows.append(self.make_sdk_row(
                    sequence, app["run_id"], "BACKEND_TRANSFER", direction, memory,
                    start + 10, end - 10, per_dpu,
                ))
                sequence += 1
        return rows

    def make_baseline_rows(self, app_rows):
        groups, totals = validator.validate_app_rows(app_rows, 2_621_440, 16, 10, 1)
        run_key = next(iter(groups))
        indexed = groups[run_key]
        output = []
        for phase in ("H2D", "D2H"):
            output.append({
                "schema_version": "upmem.va_baseline.hw.v1",
                "run_id": run_key[0],
                "repeat_id": str(run_key[1]),
                "configuration": "rank",
                "actual_dpus": "64",
                "actual_ranks": "1",
                "tasklets": "16",
                "block_size_log2": "10",
                "input_elements": "2621440",
                "scaling": "strong",
                "phase": phase,
                "duration_ns": str(totals[run_key][phase]),
                "bytes_per_dpu": str(
                    sum(int(indexed[name]["bytes_per_dpu"])
                        for name in (("PUSH_ARGS", "PUSH_A", "PUSH_B") if phase == "H2D" else ("PUSH_C",)))
                ),
                "aggregate_bytes": str(totals[run_key][f"{phase}_BYTES"]),
                "call_sequence": "test",
                "result_ok": "1",
                "expected_checksum": "42",
                "actual_checksum": "42",
                "host_numa_node": "0",
                "sdk_version": "2025.1.0",
                "source_sha256": "source",
                "host_binary_sha256": "host",
                "dpu_binary_sha256": "dpu",
                "_source_path": "baseline.csv",
            })
        return output

    def test_validates_counts_bytes_sums_and_sdk_containment(self):
        app_rows = self.make_app_rows()
        groups, totals = validator.validate_app_rows(app_rows, 2_621_440, 16, 10, 1)
        self.assertEqual(len(groups), 1)
        self.assertEqual(next(iter(totals.values()))["H2D"], 600)
        sdk_rows = self.make_sdk_rows(app_rows)
        derived = validator.validate_sdk_rows(sdk_rows, groups)
        self.assertEqual(len(derived), 3)
        for row in derived:
            self.assertEqual(
                row["c_pre_wall_ns"] + row["t_backend_wall_ns"] + row["c_tail_wall_ns"],
                96,
            )

    def test_accepts_ba_order_and_matches_sdk_by_interval(self):
        app_rows = self.make_app_rows()
        indexed = {row["component"]: row for row in app_rows}
        clock_fields = (
            "wall_start_ns", "wall_end_ns", "thread_cpu_start_ns", "thread_cpu_end_ns",
            "process_cpu_start_ns", "process_cpu_end_ns",
        )
        for left, right in (("PREPARE_A", "PREPARE_B"), ("PUSH_A", "PUSH_B")):
            for field in clock_fields:
                indexed[left][field], indexed[right][field] = indexed[right][field], indexed[left][field]
        groups, _totals = validator.validate_app_rows(app_rows, 2_621_440, 16, 10, 1)
        run_key = next(iter(groups))
        self.assertEqual(validator.detect_transfer_order(groups[run_key]), "BA")
        derived = validator.validate_sdk_rows(self.make_sdk_rows(app_rows), groups)
        self.assertEqual({row["component"] for row in derived}, {"PUSH_A", "PUSH_B", "PUSH_C"})

    def test_baseline_phase_sum_and_bytes(self):
        app_rows = self.make_app_rows()
        groups, totals = validator.validate_app_rows(app_rows, 2_621_440, 16, 10, 1)
        baseline = self.make_baseline_rows(app_rows)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=validator.BASELINE_HEADER, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(baseline)
            validator.validate_baseline_crosscheck([path], groups, totals)

    def test_rejects_corrupt_component_set(self):
        rows = self.make_app_rows()[:-1]
        with self.assertRaises(validator.ValidationError):
            validator.validate_app_rows(rows, 2_621_440, 16, 10, 1)

    def test_rejects_sdk_overflow(self):
        app_rows = self.make_app_rows()
        groups, _totals = validator.validate_app_rows(app_rows, 2_621_440, 16, 10, 1)
        sdk_rows = self.make_sdk_rows(app_rows)
        sdk_rows[0]["overflow_count"] = "1"
        with self.assertRaises(validator.ValidationError):
            validator.validate_sdk_rows(sdk_rows, groups)

    def test_rejects_backend_outside_push(self):
        app_rows = self.make_app_rows()
        groups, _totals = validator.validate_app_rows(app_rows, 2_621_440, 16, 10, 1)
        sdk_rows = self.make_sdk_rows(app_rows)
        backend = next(row for row in sdk_rows if row["event_kind"] == "BACKEND_TRANSFER")
        backend["wall_start_ns"] = "1"
        backend["wall_end_ns"] = "2"
        backend["wall_duration_ns"] = "1"
        with self.assertRaises(validator.ValidationError):
            validator.validate_sdk_rows(sdk_rows, groups)

    def test_overhead_threshold_status(self):
        stock = {(f"s{i}", i): {"H2D": 100, "D2H": 200} for i in range(4)}
        close = {(f"i{i}", i): {"H2D": 102, "D2H": 204} for i in range(4)}
        far = copy.deepcopy(close)
        for sample in far.values():
            sample["H2D"] = 110
        _rows, passed = validator.compare_overhead(stock, close, 3.0, 5.0)
        self.assertTrue(passed)
        rows, passed = validator.compare_overhead(stock, far, 3.0, 5.0)
        self.assertFalse(passed)
        self.assertIn("FAIL", {row["status"] for row in rows})

    def test_loader_rejects_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("wrong\nvalue\n")
            with self.assertRaises(validator.ValidationError):
                validator.load_csv([path], validator.APP_HEADER, validator.APP_SCHEMA)

    def test_round_trip_loaders_check_exact_headers(self):
        app_rows = self.make_app_rows()
        sdk_rows = self.make_sdk_rows(app_rows)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = (
                (root / "app.csv", validator.APP_HEADER, validator.APP_SCHEMA, app_rows),
                (root / "sdk.csv", validator.SDK_HEADER, validator.SDK_SCHEMA, sdk_rows),
            )
            for path, header, schema, rows in cases:
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=header, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(rows)
                self.assertEqual(len(validator.load_csv([path], header, schema)), len(rows))

    def test_provenance_requires_instrumented_runtime_and_hardware_backend(self):
        required = (
            "uname.txt", "lscpu.txt", "numa.txt", "numactl_show.txt", "config.txt",
            "prim_git_commit.txt", "prim_git_status.txt", "source.sha256", "binaries.sha256",
            "sdk_source_baseline_check.txt", "sdk_source.sha256", "sdk_patch_check.txt",
            "sdk_build.log", "instrumented_library.sha256", "dynamic_library_resolution.txt",
            "system_backend_libraries.txt", "system_runtime_assets.txt", "imc_status.txt",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in required:
                (root / name).write_text("evidence\n")
            (root / "sdk_source_baseline_check.txt").write_text("api/CMakeLists.txt: OK\n")
            (root / "sdk_patch_check.txt").write_text("PATCH_APPLIED=PASS\n")
            (root / "shadow_lib").mkdir()
            instrumented = root / "shadow_lib" / "libdpu.so.2025.1"
            instrumented.write_bytes(b"instrumented libdpu fixture\n")
            (root / "instrumented_library.sha256").write_text(
                f"{validator.sha256_file(instrumented)}  /tmp/{root.name}/shadow_lib/libdpu.so.2025.1\n"
            )
            (root / "dynamic_library_resolution.txt").write_text(
                f"libdpu.so.2025.1 => /tmp/{root.name}/shadow_lib/libdpu.so.2025.1\n"
                "libdpuhw.so\n"
            )
            (root / "system_backend_libraries.txt").write_text(
                "libdpuhw.so=/usr/lib/libdpuhw.so.2025.1\n"
                "libdpuhw.so.2025.1=/usr/lib/libdpuhw.so.2025.1\n"
            )
            (root / "system_runtime_assets.txt").write_text(
                "share/upmem=/usr/share/upmem\n"
            )
            result = validator.validate_provenance(root)
        self.assertEqual(result["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
