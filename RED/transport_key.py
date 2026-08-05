#!/usr/bin/env python3
"""Canonical transport keys for RED collection-transfer events."""

from __future__ import annotations

from collections.abc import Mapping


TRANSFER_OPS = {"dpu_transfer"}
ITERATIVE_SUBOPS = {"input_arguments", "input_data", "results"}
TRANSPORT_KEY_FIELDS = (
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
    "same_source_across_group",
    "phase_class",
)


def sdk_api_kind(op: str) -> str:
    return "PUSH_XFER" if op in TRANSFER_OPS else ""


def logical_distribution_class(op: str, subop: str) -> str:
    if op not in TRANSFER_OPS:
        return ""
    if subop in {"input_arguments", "input_data"}:
        return "PARTITIONED_SCATTER"
    if subop == "results":
        return "REDUCTION_GATHER"
    return "UNKNOWN"


def same_source_across_group(op: str, subop: str) -> str:
    del subop
    return "0" if op in TRANSFER_OPS else ""


def phase_class(op: str, subop: str) -> str:
    if op not in TRANSFER_OPS:
        return ""
    # The -w warmup flag describes execution context. These transfers all
    # belong to RED's repeated workload loop, so their workload phase matches.
    if subop in ITERATIVE_SUBOPS:
        return "ITERATIVE"
    return "UNKNOWN"


def transport_key(row: Mapping[str, str]) -> str:
    if row["op"] not in TRANSFER_OPS:
        return ""
    return "v2;" + ";".join(
        f"{name}={row[name]}" for name in TRANSPORT_KEY_FIELDS
    )


def transport_key_without_phase(row: Mapping[str, str]) -> str:
    if row["op"] not in TRANSFER_OPS:
        return ""
    return "v1;" + ";".join(
        f"{name}={row[name]}" for name in TRANSPORT_KEY_FIELDS
        if name != "phase_class"
    )
