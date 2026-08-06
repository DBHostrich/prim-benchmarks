#!/usr/bin/env python3

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from evaluate_transport_keys import evaluate
from transport_key import transport_key


class TransportKeyEvaluationTest(unittest.TestCase):
    def sample_row(
        self,
        repeat_id: int,
        phase: str,
        relation: str,
        duration_ns: int,
    ) -> dict[str, str]:
        row = {
            "run_id": "synthetic",
            "repeat_id": str(repeat_id),
            "event_id": "0",
            "configured_dpus": "256",
            "num_tasklets": "1",
            "op": "dpu_copy_to",
            "direction": "TO_DPU",
            "sdk_api_kind": "SINGLE_COPY",
            "logical_distribution_class": "SHARED_REPLICATION",
            "target_space": "MRAM",
            "transfer_bytes_per_dpu": "24576",
            "active_dpus": "1",
            "active_ranks": "1",
            "active_dpus_per_rank": "1",
            "rank_ordinal": "0",
            "dpu_id_in_rank": "0",
            "global_dpu_id": "0",
            "sdk_physical_rank_id": "12288",
            "sdk_slice_id": "0",
            "sdk_member_id": "0",
            "physical_dpu_identity": "rank:12288/slice:0/member:0",
            "same_source_across_group": "1",
            "phase_class": phase,
            "previous_sdk_topology_relation": relation,
            "previous_dpu_direction": "TO_DPU",
            "previous_dpu_target_relation": "SAME_REGION",
            "target_region_reuse_class": "REUSED_REGION",
            "host_numa_node": "0",
            "measured_ns": str(duration_ns),
            "transport_key": "",
        }
        row["transport_key"] = transport_key(row)
        return row

    def write_rows(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_mux_domain_predicts_phase_mixed_base_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for repeat_id in range(1, 4):
                path = root / f"trace_{repeat_id}.csv"
                self.write_rows(
                    path,
                    [
                        self.sample_row(
                            repeat_id,
                            "INIT",
                            "SAME_DPU",
                            100 + repeat_id,
                        ),
                        self.sample_row(
                            repeat_id,
                            "ITERATIVE",
                            "SAME_SLICE",
                            200 + repeat_id,
                        ),
                    ],
                )
                paths.append(path)

            summary, per_trace = evaluate(paths)
            by_key = {
                (row["model"], row["scope"]): row for row in summary
            }
            phase = by_key[("phase_v2", "BASE12_PHASE_MIXED")]
            mux_domain = by_key[("mux_domain_min", "BASE12_PHASE_MIXED")]
            base = by_key[("base12", "BASE12_PHASE_MIXED")]

            self.assertEqual(phase["coverage_pct"], "100.000000")
            self.assertEqual(mux_domain["coverage_pct"], "100.000000")
            self.assertLess(
                float(mux_domain["p90_abs_pct_error"]),
                float(base["p90_abs_pct_error"]),
            )
            self.assertEqual(len(per_trace), 42)

    def test_can_hold_out_one_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for repeat_id in range(1, 5):
                configured_dpus = "64" if repeat_id <= 2 else "128"
                path = root / f"trace_{repeat_id}.csv"
                rows = [
                    self.sample_row(
                        repeat_id,
                        "INIT",
                        "SAME_DPU",
                        100 + repeat_id,
                    ),
                    self.sample_row(
                        repeat_id,
                        "ITERATIVE",
                        "SAME_SLICE",
                        200 + repeat_id,
                    ),
                ]
                for row in rows:
                    row["configured_dpus"] = configured_dpus
                    if configured_dpus == "128":
                        row["rank_ordinal"] = "1"
                self.write_rows(path, rows)
                paths.append(path)

            summary, per_holdout = evaluate(paths, "configuration")
            domain = next(
                row
                for row in summary
                if row["model"] == "mux_domain_hierarchical"
                and row["scope"] == "BASE12_PHASE_MIXED"
            )
            self.assertEqual(domain["holdout_unit"], "configuration")
            self.assertEqual(domain["holdout_groups"], 2)
            self.assertEqual(domain["coverage_pct"], "100.000000")
            self.assertEqual(domain["fallback_predicted_pct"], "100.000000")
            self.assertEqual(len(per_holdout), 28)


if __name__ == "__main__":
    unittest.main()
