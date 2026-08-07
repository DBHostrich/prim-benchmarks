#!/usr/bin/env python3
"""Canonical v8 keys for GEMV collection transfers."""

from __future__ import annotations

from collections.abc import Mapping


TRANSFER_OPS = {"dpu_push_xfer"}
V8_FIELDS = (
    "op",
    "direction",
    "sdk_api_kind",
    "timing_scope",
    "logical_distribution_class",
    "target_space",
    "transfer_bytes_per_dpu",
    "active_dpus",
    "active_ranks",
    "active_dpus_per_rank",
    "same_source_across_group",
    "host_numa_node",
    "dpu_rank_numa_nodes",
    "cpu_dpu_numa_relation",
    "dpu_channel_ids",
    "dpu_sysfs_rank_ids",
    "dpu_ci_ids",
    "dpu_member_ids",
    "allocated_topology_signature",
    "allocated_dpus",
    "allocated_ranks",
    "previous_sdk_mux_domain_class",
)


def sdk_api_kind(op: str) -> str:
    return "PUSH_XFER" if op in TRANSFER_OPS else ""


def logical_distribution_class(op: str, subop: str) -> str:
    if op not in TRANSFER_OPS:
        return ""
    if subop == "input_vector":
        return "SHARED_REPLICATION"
    if subop in {"input_arguments", "input_matrix"}:
        return "PARTITIONED_SCATTER"
    if subop == "output_vector":
        return "PARTITIONED_GATHER"
    return "UNKNOWN"


def same_source_across_group(op: str, subop: str) -> str:
    if op not in TRANSFER_OPS:
        return ""
    return "1" if subop == "input_vector" else "0"


def phase_class(op: str, warmup: str) -> str:
    if op not in TRANSFER_OPS:
        return ""
    return "WARMUP" if warmup == "1" else "ITERATIVE"


def transport_key(row: Mapping[str, str]) -> str:
    if row["op"] not in TRANSFER_OPS:
        return ""
    return "v8;" + ";".join(f"{field}={row[field]}" for field in V8_FIELDS)
