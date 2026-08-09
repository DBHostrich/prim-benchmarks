from __future__ import annotations

import unittest

from transport_key import phase_class, transport_key, transport_key_v8


def transfer_row() -> dict[str, str]:
    return {
        "op": "dpu_push_xfer",
        "direction": "TO_DPU",
        "sdk_api_kind": "PUSH_XFER",
        "timing_scope": "PUSH_ONLY",
        "logical_distribution_class": "SHARED_REPLICATION",
        "target_space": "MRAM",
        "transfer_bytes_per_dpu": "32768",
        "active_dpus": "128",
        "active_ranks": "2",
        "active_dpus_per_rank": "64|64",
        "same_source_across_group": "1",
        "host_numa_node": "0",
        "dpu_rank_numa_nodes": "0|0",
        "cpu_dpu_numa_relation": "LOCAL",
        "dpu_channel_ids": "1|2",
        "dpu_sysfs_rank_ids": "0|4",
        "dpu_ci_ids": "0-7",
        "dpu_member_ids": "0-7",
        "allocated_topology_signature": "r0@n0@c1|r4@n0@c2",
        "allocated_dpus": "128",
        "allocated_ranks": "2",
        "previous_sdk_mux_domain_class": "COLLECTION",
        "previous_sdk_op_class": "PUSH_XFER_TO_DPU_MRAM",
        "source_buffer_reuse_class": "FIRST_USE",
        "target_region_reuse_class": "FIRST_ACCESS",
    }


class TransportKeyTests(unittest.TestCase):
    def test_collection_key_contains_physical_allocation(self) -> None:
        key = transport_key(transfer_row())
        self.assertTrue(key.startswith("v9;op=dpu_push_xfer;"))
        self.assertIn("timing_scope=PUSH_ONLY", key)
        self.assertIn("dpu_sysfs_rank_ids=0|4", key)
        self.assertIn("allocated_dpus=128", key)
        self.assertIn("previous_sdk_mux_domain_class=COLLECTION", key)
        self.assertIn(
            "previous_sdk_op_class=PUSH_XFER_TO_DPU_MRAM", key
        )
        self.assertIn("source_buffer_reuse_class=FIRST_USE", key)

    def test_v8_key_can_be_reconstructed_for_comparison(self) -> None:
        key = transport_key_v8(transfer_row())
        self.assertTrue(key.startswith("v8;op=dpu_push_xfer;"))
        self.assertNotIn("previous_sdk_op_class", key)

    def test_warmup_is_diagnostic_outside_v8_key(self) -> None:
        row = transfer_row()
        row["phase_class"] = "WARMUP"
        warmup_key = transport_key(row)
        row["phase_class"] = "ITERATIVE"
        self.assertEqual(transport_key(row), warmup_key)
        self.assertEqual(phase_class(row["op"], "1"), "WARMUP")
        self.assertEqual(phase_class(row["op"], "0"), "ITERATIVE")


if __name__ == "__main__":
    unittest.main()
