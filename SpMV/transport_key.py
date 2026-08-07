#!/usr/bin/env python3
"""Canonical transport keys for SpMV host transfer events."""

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
PHYSICAL_TRANSPORT_SHAPE_FIELDS = tuple(
    field
    for field in BASE_TRANSPORT_KEY_FIELDS
    if field not in {"rank_ordinal", "dpu_id_in_rank"}
)
CPU_DPU_PHYSICAL_TOPOLOGY_FIELDS = (
    "host_numa_node",
    "dpu_rank_numa_node",
    "cpu_dpu_numa_relation",
    "dpu_channel_id",
    "dpu_sysfs_rank_id",
    "dpu_ci_id",
    "dpu_member_id",
)
PHASE_TRANSPORT_KEY_FIELDS = BASE_TRANSPORT_KEY_FIELDS + ("phase_class",)
SHARED_SOURCE_SUBOPS = {"input_vector"}
INIT_SUBOPS = {"row_ptrs", "nonzeros", "input_vector", "params"}
FINALIZE_SUBOPS = {"output_vector"}


def sdk_api_kind(op: str) -> str:
    return "SINGLE_COPY" if op in TRANSFER_OPS else ""


def logical_distribution_class(op: str, subop: str) -> str:
    if op == "dpu_copy_to":
        if subop in SHARED_SOURCE_SUBOPS:
            return "SHARED_REPLICATION"
        return "PARTITIONED_SCATTER"
    if op == "dpu_copy_from":
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
    if subop in FINALIZE_SUBOPS:
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
        f"numa:{row['dpu_rank_numa_node']}/"
        f"channel:{row['dpu_channel_id']}/"
        f"rank:{row['dpu_sysfs_rank_id']}/"
        f"ci:{row['dpu_ci_id']}/member:{row['dpu_member_id']}"
    )


def mux_domain_class(relation: str) -> str:
    if relation in {"SAME_DPU", "SAME_MUX_PAIR"}:
        return "SAME_MUX_DOMAIN"
    if relation in {"SAME_SLICE", "SAME_RANK", "OTHER_RANK"}:
        return "DIFFERENT_MUX_DOMAIN"
    return relation


def allocated_topology_context(row: Mapping[str, str]) -> str:
    return (
        f"allocated_dpus={row['configured_dpus']};"
        f"allocated_ranks={row['actual_ranks']}"
    )


def transport_key(row: Mapping[str, str]) -> str:
    if row["op"] not in TRANSFER_OPS:
        return ""
    fields = PHYSICAL_TRANSPORT_SHAPE_FIELDS + CPU_DPU_PHYSICAL_TOPOLOGY_FIELDS
    base = ";".join(f"{name}={row[name]}" for name in fields)
    domain = mux_domain_class(row["previous_sdk_topology_relation"])
    return (
        f"v8;{base};{allocated_topology_context(row)};"
        f"previous_sdk_mux_domain_class={domain}"
    )


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


def sdk_topology_relation(
    previous: Mapping[str, str] | None,
    current: Mapping[str, str],
) -> str:
    if previous is None:
        return "NONE"
    if previous.get("global_dpu_id", "") == "":
        return "COLLECTION"
    if previous["global_dpu_id"] == current["global_dpu_id"]:
        return "SAME_DPU"
    previous_rank = previous.get("sdk_physical_rank_id", "")
    current_rank = current.get("sdk_physical_rank_id", "")
    if previous_rank and current_rank and previous_rank != current_rank:
        return "OTHER_RANK"
    if (
        (not previous_rank or not current_rank)
        and previous["rank_ordinal"] != current["rank_ordinal"]
    ):
        return "OTHER_RANK"
    if previous["sdk_slice_id"] != current["sdk_slice_id"]:
        return "SAME_RANK"
    if int(previous["sdk_member_id"]) // 2 == int(current["sdk_member_id"]) // 2:
        return "SAME_MUX_PAIR"
    return "SAME_SLICE"


def derive_hardware_contexts(
    rows: list[Mapping[str, str]],
) -> list[dict[str, str]]:
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
                max(0, int(row["host_start_ns"]) - int(previous_sdk["host_end_ns"]))
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
            "host_buffer_page_offset": str(int(row["host_buffer_address"]) % 4096),
            "host_buffer_reuse_class": host_reuse,
        }
        last_dpu_row[dpu_id] = index
        last_dpu_launch_count[dpu_id] = launch_count

    return contexts
