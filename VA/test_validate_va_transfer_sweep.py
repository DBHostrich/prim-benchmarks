import sys
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_va_transfer_breakdown as base
import validate_va_transfer_sweep as sweep


class ValidateVaTransferSweepTest(unittest.TestCase):
    def make_plan(self):
        rows = []
        for size in (8192, 16384):
            bytes_per_dpu = sweep.expected_bytes_per_dpu(size)
            for rep in range(1, 5):
                order = "AB" if rep % 2 else "BA"
                rows.append({
                    "kind": "formal", "input_elements": str(size),
                    "bytes_per_dpu": str(bytes_per_dpu), "transfer_order": order,
                    "replicate": str(rep), "library_mode": "instrumented",
                    "run_id": f"formal_n{size}_{order.lower()}_{rep:02d}",
                })
            for rep in range(1, 3):
                order = "AB" if rep % 2 else "BA"
                rows.append({
                    "kind": "warmup", "input_elements": str(size),
                    "bytes_per_dpu": str(bytes_per_dpu), "transfer_order": order,
                    "replicate": str(rep), "library_mode": "instrumented",
                    "run_id": f"warmup_n{size}_{order.lower()}_{rep:02d}",
                })
        for library_mode in ("stock", "instrumented"):
            for rep in range(1, 3):
                order = "AB" if rep % 2 else "BA"
                rows.append({
                    "kind": "overhead", "input_elements": "8192",
                    "bytes_per_dpu": str(sweep.expected_bytes_per_dpu(8192)),
                    "transfer_order": order, "replicate": str(rep),
                    "library_mode": library_mode,
                    "run_id": f"overhead_{library_mode}_n8192_{order.lower()}_{rep:02d}",
                })
        return rows

    def test_plan_accepts_balanced_orders(self):
        plan = sweep.validate_plan(self.make_plan(), [8192, 16384], [8192], 4, 2, 2)
        self.assertEqual(len(plan), 16)

    def test_plan_rejects_formal_order_imbalance(self):
        rows = self.make_plan()
        formal = next(row for row in rows if row["kind"] == "formal" and row["transfer_order"] == "BA")
        formal["transfer_order"] = "AB"
        with self.assertRaises(base.ValidationError):
            sweep.validate_plan(rows, [8192, 16384], [8192], 4, 2, 2)

    def test_sample_summary_preserves_order_and_position(self):
        derived = [{
            "run_id": "formal_n8192_ba_02", "component": "PUSH_B", "direction": "H2D",
            "bytes_per_dpu": 512, "aggregate_bytes": 32768,
            "c_pre_wall_ns": 10, "t_backend_wall_ns": 80, "c_tail_wall_ns": 10,
            "backend_process_cpu_ns": 240,
        }]
        plan = {
            "formal_n8192_ba_02": {
                "input_elements": "8192", "transfer_order": "BA",
            }
        }
        samples = sweep.make_samples(derived, plan)
        self.assertEqual(samples[0]["execution_position"], "FIRST")
        self.assertEqual(samples[0]["backend_share_pct"], "80.000000000")
        summary = sweep.summarize_samples(samples)
        self.assertEqual(summary[0]["samples"], 1)
        self.assertEqual(summary[0]["cpu_parallelism_p50"], "3.000000")

    def test_baseline_sequence_follows_ba_order(self):
        rows = [{
            "run_id": "formal_n8192_ba_02",
            "phase": "H2D",
            "call_sequence": "prepare_args>push_args>prepare_B>push_B>prepare_A>push_A",
        }]
        plan = {"formal_n8192_ba_02": {
            "kind": "formal", "library_mode": "instrumented", "transfer_order": "BA",
        }}
        sweep.validate_baseline_orders(rows, plan)

    def test_baseline_sequence_rejects_order_mismatch(self):
        rows = [{
            "run_id": "formal_n8192_ba_02",
            "phase": "H2D",
            "call_sequence": "prepare_args>push_args>prepare_A>push_A>prepare_B>push_B",
        }]
        plan = {"formal_n8192_ba_02": {
            "kind": "formal", "library_mode": "instrumented", "transfer_order": "BA",
        }}
        with self.assertRaises(base.ValidationError):
            sweep.validate_baseline_orders(rows, plan)

    def test_strict_overhead_is_opt_in(self):
        arguments = [
            "validate_va_transfer_sweep.py",
            "--result-root", "/tmp/result",
            "--input-elements", "8192",
            "--overhead-input-elements", "8192",
            "--output-dir", "/tmp/output",
        ]
        with mock.patch.object(sys, "argv", arguments):
            self.assertFalse(sweep.parse_args().strict_overhead)
        with mock.patch.object(sys, "argv", arguments + ["--strict-overhead"]):
            self.assertTrue(sweep.parse_args().strict_overhead)
        self.assertFalse(sweep.overhead_requires_failure(False, False))
        self.assertTrue(sweep.overhead_requires_failure(False, True))
        self.assertFalse(sweep.overhead_requires_failure(True, True))


if __name__ == "__main__":
    unittest.main()
