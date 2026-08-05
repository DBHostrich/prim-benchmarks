#!/usr/bin/env python3
"""Canonical measurement labels for BFS host events."""

from __future__ import annotations

from collections.abc import Mapping


def api_type(op: str) -> str:
    if op in {"dpu_copy_to", "dpu_copy_from"}:
        return "single_copy"
    if op == "dpu_launch":
        return "collection_sync"
    return "collection"


def logical_distribution_class(op: str, subop: str) -> str:
    if op == "dpu_copy_to":
        if subop in {"visited_init", "frontier_init", "frontier_broadcast"}:
            return "SHARED_REPLICATION"
        return "PARTITIONED_SCATTER"
    if op == "dpu_copy_from":
        if subop == "frontier_result":
            return "REDUCTION_GATHER"
        return "PARTITIONED_GATHER"
    return "none"


def same_source_across_group(op: str, subop: str) -> str:
    if op == "dpu_copy_to":
        return str(
            int(
                subop
                in {"visited_init", "frontier_init", "frontier_broadcast"}
            )
        )
    if op == "dpu_copy_from":
        return "0"
    return "none"


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


def measurement_label(row: Mapping[str, str]) -> str:
    has_dpu = row["global_dpu_id"] != ""
    direction = row["direction"] or "none"
    target_space = row["target_space"] if has_dpu else "none"
    transfer_bytes = row["transfer_bytes"] if has_dpu else "0"
    rank = row["rank_ordinal"] if has_dpu else "all"
    dpu = row["dpu_id_in_rank"] if has_dpu else "all"
    return (
        f"v2;op={row['op']};dir={direction};api={row['api_type']};"
        f"dist={row['logical_distribution_class']};space={target_space};"
        f"bytes={transfer_bytes};rank={rank};dpu={dpu};"
        f"same_source={row['same_source_across_group']};"
        f"addr={row['offset_feature']};call={row['call_context']};"
        f"numa={row['host_numa_node']}"
    )
