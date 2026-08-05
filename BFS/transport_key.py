#!/usr/bin/env python3
"""Canonical transport keys for BFS host transfer events."""

from __future__ import annotations

from collections.abc import Mapping


TRANSFER_OPS = {"dpu_copy_to", "dpu_copy_from"}
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
SHARED_SOURCE_SUBOPS = {
    "visited_init",
    "frontier_init",
    "frontier_broadcast",
}
INIT_SUBOPS = {
    "node_ptrs",
    "neighbor_idxs",
    "node_level_init",
    "visited_init",
    "frontier_init",
    "params_init",
}
ITERATIVE_SUBOPS = {
    "frontier_result",
    "frontier_broadcast",
    "params_level",
}


def sdk_api_kind(op: str) -> str:
    if op in TRANSFER_OPS:
        return "SINGLE_COPY"
    return ""


def logical_distribution_class(op: str, subop: str) -> str:
    if op == "dpu_copy_to":
        if subop in SHARED_SOURCE_SUBOPS:
            return "SHARED_REPLICATION"
        return "PARTITIONED_SCATTER"
    if op == "dpu_copy_from":
        # Every BFS DPU returns a full frontier bitmap. The host OR-reduces
        # those overlapping logical results, so this is not a partitioned
        # gather into disjoint host slices.
        if subop == "frontier_result":
            return "REDUCTION_GATHER"
        return "PARTITIONED_GATHER"
    return ""


def same_source_across_group(op: str, subop: str) -> str:
    if op == "dpu_copy_to":
        return str(int(subop in SHARED_SOURCE_SUBOPS))
    if op == "dpu_copy_from":
        return "0"
    return ""


def phase_class(op: str, subop: str) -> str:
    if op not in TRANSFER_OPS:
        return ""
    if subop in INIT_SUBOPS:
        return "INIT"
    if subop in ITERATIVE_SUBOPS:
        return "ITERATIVE"
    if subop == "node_level_result":
        return "FINALIZE"
    return "UNKNOWN"


def offset_feature(row: Mapping[str, str]) -> str:
    if row["global_dpu_id"] == "":
        return "none"
    offset = int(row["offset_bytes"])
    transfer_bytes = int(row["transfer_bytes"])
    pages = (
        0
        if transfer_bytes == 0
        else ((offset % 4096) + transfer_bytes + 4095) // 4096
    )
    return (
        f"off={offset}:a8={int(offset % 8 == 0)}:a64={int(offset % 64 == 0)}:"
        f"p4k={offset // 4096}:pages={pages}"
    )


def call_context(row: Mapping[str, str]) -> str:
    dpu_index = row["dpu_op_call_index"] or "none"
    return (
        f"opidx={row['op_call_index']}:dpuopidx={dpu_index}:"
        f"process={row['process_state']}:prewarm={row['pretrace_warmup_runs']}"
    )


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
