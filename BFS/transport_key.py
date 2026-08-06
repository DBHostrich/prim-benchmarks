#!/usr/bin/env python3
"""Canonical transport keys for BFS host transfer events."""

from __future__ import annotations

from collections.abc import Mapping


TRANSFER_OPS = {"dpu_copy_to", "dpu_copy_from"}
BASE_TRANSPORT_KEY_FIELDS = (
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
)
PHASE_TRANSPORT_KEY_FIELDS = BASE_TRANSPORT_KEY_FIELDS + ("phase_class",)
MUX_RELATION_MIN_FIELDS = BASE_TRANSPORT_KEY_FIELDS + (
    "previous_sdk_topology_relation",
)
RANK_INVARIANT_BASE_FIELDS = tuple(
    field for field in BASE_TRANSPORT_KEY_FIELDS if field != "rank_ordinal"
)
HISTORY_MIN_FIELDS = (
    "sdk_slice_id",
    "sdk_member_id",
    "previous_dpu_direction",
    "previous_dpu_target_relation",
    "target_region_reuse_class",
    "host_numa_node",
)
PHYSICAL_DPU_IDENTITY_FIELDS = (
    "sdk_physical_rank_id",
    "sdk_slice_id",
    "sdk_member_id",
)
FULL_HARDWARE_CONTEXT_FIELDS = (
    "sdk_slice_id",
    "sdk_member_id",
    "previous_dpu_direction",
    "previous_dpu_transfer_bytes",
    "previous_dpu_target_relation",
    "launches_since_previous_dpu_transfer",
    "target_region_reuse_class",
    "host_buffer_page_offset",
    "host_buffer_reuse_class",
    "host_numa_node",
)
V5_TRANSPORT_KEY_FIELDS = (
    BASE_TRANSPORT_KEY_FIELDS
    + PHYSICAL_DPU_IDENTITY_FIELDS
    + HISTORY_MIN_FIELDS[2:]
)
TRANSPORT_KEY_FIELDS = (
    BASE_TRANSPORT_KEY_FIELDS
    + PHYSICAL_DPU_IDENTITY_FIELDS
    + ("previous_sdk_topology_relation",)
    + HISTORY_MIN_FIELDS[2:]
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


def physical_dpu_identity(row: Mapping[str, str]) -> str:
    if row["global_dpu_id"] == "":
        return ""
    return (
        f"rank:{row['sdk_physical_rank_id']}/"
        f"slice:{row['sdk_slice_id']}/member:{row['sdk_member_id']}"
    )


def transport_key(row: Mapping[str, str]) -> str:
    if row["op"] not in TRANSFER_OPS:
        return ""
    return "v6;" + ";".join(
        f"{name}={row[name]}" for name in TRANSPORT_KEY_FIELDS
    )


def transport_key_without_mux_pair_context(row: Mapping[str, str]) -> str:
    if row["op"] not in TRANSFER_OPS:
        return ""
    return "v5;" + ";".join(
        f"{name}={row[name]}" for name in V5_TRANSPORT_KEY_FIELDS
    )


def transport_key_without_physical_rank(row: Mapping[str, str]) -> str:
    if row["op"] not in TRANSFER_OPS:
        return ""
    fields = BASE_TRANSPORT_KEY_FIELDS + HISTORY_MIN_FIELDS
    return "v4;" + ";".join(f"{name}={row[name]}" for name in fields)


def transport_key_with_full_context(row: Mapping[str, str]) -> str:
    if row["op"] not in TRANSFER_OPS:
        return ""
    fields = BASE_TRANSPORT_KEY_FIELDS + FULL_HARDWARE_CONTEXT_FIELDS
    return "v3;" + ";".join(f"{name}={row[name]}" for name in fields)


def transport_key_with_phase(row: Mapping[str, str]) -> str:
    if row["op"] not in TRANSFER_OPS:
        return ""
    return "v2;" + ";".join(
        f"{name}={row[name]}" for name in PHASE_TRANSPORT_KEY_FIELDS
    )


def transport_key_without_phase(row: Mapping[str, str]) -> str:
    if row["op"] not in TRANSFER_OPS:
        return ""
    return "v1;" + ";".join(
        f"{name}={row[name]}" for name in BASE_TRANSPORT_KEY_FIELDS
    )


def mux_domain_class(relation: str) -> str:
    if relation in {"SAME_DPU", "SAME_MUX_PAIR"}:
        return "SAME_MUX_DOMAIN"
    if relation in {"SAME_SLICE", "SAME_RANK", "OTHER_RANK"}:
        return "DIFFERENT_MUX_DOMAIN"
    return relation


def transport_key_with_mux_relation_min(row: Mapping[str, str]) -> str:
    """Base transport shape plus the exact immediate SDK topology relation."""
    if row["op"] not in TRANSFER_OPS:
        return ""
    return "mux_relation_min;" + ";".join(
        f"{name}={row[name]}" for name in MUX_RELATION_MIN_FIELDS
    )


def transport_key_with_mux_domain_min(row: Mapping[str, str]) -> str:
    """Base transport shape plus compressed immediate MUX-domain state."""
    if row["op"] not in TRANSFER_OPS:
        return ""
    base = ";".join(
        f"{name}={row[name]}" for name in BASE_TRANSPORT_KEY_FIELDS
    )
    domain = mux_domain_class(row["previous_sdk_topology_relation"])
    return f"mux_domain_min;{base};previous_sdk_mux_domain_class={domain}"


def transport_key_with_mux_domain_rank_invariant(
    row: Mapping[str, str],
) -> str:
    """MUX-domain key that shares one table across allocation-local ranks."""
    if row["op"] not in TRANSFER_OPS:
        return ""
    base = ";".join(
        f"{name}={row[name]}" for name in RANK_INVARIANT_BASE_FIELDS
    )
    domain = mux_domain_class(row["previous_sdk_topology_relation"])
    return (
        "mux_domain_rank_invariant;"
        f"{base};previous_sdk_mux_domain_class={domain}"
    )


def allocated_topology_context(row: Mapping[str, str]) -> str:
    return (
        f"allocated_dpus={row['configured_dpus']};"
        f"allocated_ranks={row['actual_ranks']}"
    )


def transport_key_with_phase_allocated_topology(
    row: Mapping[str, str],
) -> str:
    if row["op"] not in TRANSFER_OPS:
        return ""
    base = ";".join(
        f"{name}={row[name]}" for name in BASE_TRANSPORT_KEY_FIELDS
    )
    return (
        f"phase_allocated_topology;{base};"
        f"{allocated_topology_context(row)};phase_class={row['phase_class']}"
    )


def transport_key_with_mux_domain_allocated_topology(
    row: Mapping[str, str],
) -> str:
    if row["op"] not in TRANSFER_OPS:
        return ""
    base = ";".join(
        f"{name}={row[name]}" for name in BASE_TRANSPORT_KEY_FIELDS
    )
    domain = mux_domain_class(row["previous_sdk_topology_relation"])
    return (
        f"mux_domain_allocated_topology;{base};"
        f"{allocated_topology_context(row)};"
        f"previous_sdk_mux_domain_class={domain}"
    )


def sdk_topology_relation(
    previous: Mapping[str, str] | None,
    current: Mapping[str, str],
) -> str:
    if previous is None:
        return "NONE"
    previous_dpu_id = previous.get("global_dpu_id", "")
    if previous_dpu_id == "":
        return "COLLECTION"
    if previous_dpu_id == current["global_dpu_id"]:
        return "SAME_DPU"
    if previous["rank_ordinal"] != current["rank_ordinal"]:
        return "OTHER_RANK"
    if previous["sdk_slice_id"] != current["sdk_slice_id"]:
        return "SAME_RANK"
    if int(previous["sdk_member_id"]) // 2 == int(current["sdk_member_id"]) // 2:
        return "SAME_MUX_PAIR"
    return "SAME_SLICE"


def derive_hardware_contexts(
    rows: list[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Reconstruct hardware-history fields from one ordered trace."""
    contexts: list[dict[str, str]] = [{} for _ in rows]
    last_dpu_row: dict[str, int] = {}
    last_dpu_launch_count: dict[str, int] = {}
    seen_regions: set[tuple[str, str, str]] = set()
    host_buffers: dict[tuple[str, str], str] = {}
    launch_count = 0

    for index, row in enumerate(rows):
        if row["op"] == "dpu_launch":
            launch_count += 1
        if row["op"] not in TRANSFER_OPS:
            continue

        dpu_id = row["global_dpu_id"]
        previous_sdk = rows[index - 1] if index else None
        if previous_sdk is None:
            previous_sdk_op = "NONE"
            previous_sdk_direction = "NONE"
            previous_sdk_bytes = "0"
            previous_sdk_relation = "NONE"
            since_previous_sdk = "0"
        else:
            previous_sdk_op = previous_sdk["op"]
            previous_sdk_direction = previous_sdk["direction"] or "NONE"
            previous_sdk_bytes = (
                previous_sdk["transfer_bytes"]
                if previous_sdk["op"] in TRANSFER_OPS
                else "0"
            )
            previous_sdk_relation = sdk_topology_relation(previous_sdk, row)
            since_previous_sdk = str(
                max(
                    0,
                    int(row["host_start_ns"])
                    - int(previous_sdk["host_end_ns"]),
                )
            )

        previous_index = last_dpu_row.get(dpu_id)
        if previous_index is None:
            previous_dpu_direction = "NONE"
            previous_dpu_bytes = "0"
            previous_target_relation = "NONE"
        else:
            previous_dpu = rows[previous_index]
            previous_dpu_direction = previous_dpu["direction"]
            previous_dpu_bytes = previous_dpu["transfer_bytes"]
            same_region = (
                previous_dpu["offset_bytes"] == row["offset_bytes"]
                and previous_dpu["transfer_bytes"] == row["transfer_bytes"]
            )
            previous_target_relation = (
                "SAME_REGION" if same_region else "DIFFERENT_REGION"
            )

        region = (dpu_id, row["offset_bytes"], row["transfer_bytes"])
        target_reuse = (
            "REUSED_REGION" if region in seen_regions else "FIRST_REGION_ACCESS"
        )
        seen_regions.add(region)

        host_key = (row["host_buffer_address"], row["transfer_bytes"])
        prior_host_direction = host_buffers.get(host_key)
        if prior_host_direction is None:
            host_reuse = "FIRST_SDK_USE"
        elif prior_host_direction == row["direction"]:
            host_reuse = "SAME_DIRECTION_REUSE"
        else:
            host_reuse = "DIRECTION_SWITCH_REUSE"
        host_buffers[host_key] = row["direction"]

        contexts[index] = {
            "previous_sdk_op": previous_sdk_op,
            "previous_sdk_direction": previous_sdk_direction,
            "previous_sdk_transfer_bytes": previous_sdk_bytes,
            "previous_sdk_topology_relation": previous_sdk_relation,
            "ns_since_previous_sdk_event": since_previous_sdk,
            "previous_dpu_direction": previous_dpu_direction,
            "previous_dpu_transfer_bytes": previous_dpu_bytes,
            "previous_dpu_target_relation": previous_target_relation,
            "launches_since_previous_dpu_transfer": str(
                launch_count - last_dpu_launch_count.get(dpu_id, 0)
            ),
            "target_region_reuse_class": target_reuse,
            "host_buffer_page_offset": str(
                int(row["host_buffer_address"]) % 4096
            ),
            "host_buffer_reuse_class": host_reuse,
        }
        last_dpu_row[dpu_id] = index
        last_dpu_launch_count[dpu_id] = launch_count

    return contexts
