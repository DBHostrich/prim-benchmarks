import csv
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_va_baseline_trace as validator


class ValidateVaBaselineTraceTest(unittest.TestCase):
    def make_rows(self, config: str, dpus: int, repeat: int):
        durations = {
            "SETUP": 101,
            "CPU_REFERENCE": 102,
            "H2D": 103,
            "KERNEL": 104,
            "D2H": 105,
            "VERIFY": 106,
            "TOTAL": 312,
        }
        rows = []
        for phase in validator.PHASES:
            per_dpu, aggregate = validator.expected_bytes(phase, 2_621_440, dpus)
            rows.append({
                "schema_version": validator.SCHEMA_VERSION,
                "run_id": f"VA_{config}",
                "repeat_id": str(repeat),
                "configuration": config,
                "actual_dpus": str(dpus),
                "actual_ranks": "1",
                "tasklets": "16",
                "block_size_log2": "10",
                "input_elements": "2621440",
                "scaling": "strong",
                "phase": phase,
                "duration_ns": str(durations[phase]),
                "bytes_per_dpu": str(per_dpu),
                "aggregate_bytes": str(aggregate),
                "call_sequence": validator.EXPECTED_CALLS[phase],
                "result_ok": "1",
                "expected_checksum": "42",
                "actual_checksum": "42",
                "host_numa_node": "0",
                "sdk_version": "2024.2.0",
                "source_sha256": "source",
                "host_binary_sha256": "host",
                "dpu_binary_sha256": "dpu",
            })
        return rows

    def test_accepts_complete_two_configuration_trace(self):
        rows = []
        for repeat in (1, 2):
            rows.extend(self.make_rows("single", 1, repeat))
            rows.extend(self.make_rows("rank", 64, repeat))
        manifest = validator.validate_rows(rows, 2, 2_621_440, 16, 10)
        self.assertEqual(manifest["status"], "PASS")
        self.assertEqual(manifest["configurations"]["rank"]["actual_dpus"], 64)

    def test_rejects_total_mismatch(self):
        rows = self.make_rows("single", 1, 1) + self.make_rows("rank", 64, 1)
        for row in rows:
            if row["configuration"] == "rank" and row["phase"] == "TOTAL":
                row["duration_ns"] = "313"
        with self.assertRaises(validator.ValidationError):
            validator.validate_rows(rows, 1, 2_621_440, 16, 10)

    def test_round_trip_loader_checks_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            rows = self.make_rows("single", 1, 1)
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=validator.HEADER)
                writer.writeheader()
                writer.writerows(rows)
            self.assertEqual(len(validator.load_rows([path])), len(validator.PHASES))

    def test_provenance_accepts_clean_git_status(self):
        required = (
            "uname.txt", "lscpu.txt", "numa.txt", "numactl_show.txt",
            "prim_git_commit.txt", "prim_git_status.txt", "dpu_compiler_version.txt",
            "dpu_sdk_flags.txt", "source.sha256", "binaries.sha256", "config.txt",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in required:
                content = "" if name == "prim_git_status.txt" else "evidence\n"
                (root / name).write_text(content)
            (root / "config.txt").write_text(
                "INPUT_ELEMENTS=2621440\nTASKLETS=16\nBLOCK_SIZE_LOG2=10\n"
                "SCALING=strong\nVA_VALIDATION_INPUT=1\n"
            )
            result = validator.validate_provenance(root)
        self.assertEqual(result["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
