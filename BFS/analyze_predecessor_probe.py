#!/usr/bin/env python3
"""Validate and analyze the BFS 2x2 predecessor factorial probe."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_group_context_probe import distribution, effect_class, quantile


CONDITIONS = (
    "SAME_DPU_24576B_PREDECESSOR",
    "SAME_DPU_48B_PREDECESSOR",
    "PAIRED_DPU_24576B_PREDECESSOR",
    "PAIRED_DPU_48B_PREDECESSOR",
)
COMPARISONS = (
    (
        "PREDECESSOR_SIZE_EFFECT_SAME_DPU",
        "SAME_DPU_24576B_PREDECESSOR",
        "SAME_DPU_48B_PREDECESSOR",
    ),
    (
        "PREDECESSOR_SIZE_EFFECT_PAIRED_DPU",
        "PAIRED_DPU_24576B_PREDECESSOR",
        "PAIRED_DPU_48B_PREDECESSOR",
    ),
    (
        "PREDECESSOR_LOCALITY_EFFECT_24576B",
        "PAIRED_DPU_24576B_PREDECESSOR",
        "SAME_DPU_24576B_PREDECESSOR",
    ),
    (
        "PREDECESSOR_LOCALITY_EFFECT_48B",
        "PAIRED_DPU_48B_PREDECESSOR",
        "SAME_DPU_48B_PREDECESSOR",
    ),
)
REQUIRED_FIELDS = {
    "run_id",
    "process_repeat",
    "configured_dpus",
    "num_tasklets",
    "actual_ranks",
    "sample_index",
    "target_order_seed",
    "target_visit_position",
    "call_position_in_rank",
    "condition_order_index",
    "condition",
    "target_global_dpu_id",
    "rank_ordinal",
    "dpu_id_in_rank",
    "dpu_ordinal_in_rank",
    "sdk_slice_id",
    "sdk_member_id",
    "ordinal_parity",
    "predecessor_global_dpu_id",
    "predecessor_rank_ordinal",
    "predecessor_dpu_ordinal_in_rank",
    "predecessor_slice_id",
    "predecessor_member_id",
    "transition_class",
    "op",
    "direction",
    "sdk_api_kind",
    "logical_distribution_class",
    "target_space",
    "transfer_bytes_per_dpu",
    "active_dpus",
    "active_ranks",
    "active_dpus_per_rank",
    "offset_bytes",
    "same_source_across_group",
    "phase_class",
    "source_buffer_class",
    "same_source_across_conditions",
    "source_pointer",
    "source_content_hash",
    "source_alignment_bytes",
    "target_precondition",
    "previous_op",
    "previous_direction",
    "previous_transfer_bytes",
    "previous_offset_bytes",
    "same_dpu_as_previous",
    "same_rank_as_previous",
    "precondition_start_ns",
    "precondition_end_ns",
    "precondition_ns",
    "source_pretouch_ns",
    "predecessor_start_ns",
    "predecessor_end_ns",
    "predecessor_ns",
    "ns_since_previous_sdk_event",
    "host_start_ns",
    "host_end_ns",
    "measured_ns",
    "verification",
}


def expected_condition_semantics(condition: str) -> tuple[bool, int]:
    same_dpu = condition.startswith("SAME_DPU_")
    predecessor_bytes = 24576 if "24576B" in condition else 48
    return same_dpu, predecessor_bytes


def position_class(position: int, rank_size: int = 64) -> str:
    if position == 0:
        return "FIRST"
    if position < 8:
        return "EARLY"
    if position < rank_size - 8:
        return "MIDDLE"
    if position < rank_size - 1:
        return "LATE"
    return "LAST"


def read_and_validate(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_trace_keys: set[tuple[str, str]] = set()
    hashes_by_size: dict[str, set[str]] = defaultdict(set)
    topology_reference: dict[str, tuple[str, ...]] = {}
    predecessor_offset_reference: dict[str, str] = {}

    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_FIELDS - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path}: missing fields {sorted(missing)}")
            file_rows = list(reader)
        if not file_rows:
            raise ValueError(f"{path}: empty trace")

        trace_key = (file_rows[0]["run_id"], file_rows[0]["process_repeat"])
        if trace_key in seen_trace_keys:
            raise ValueError(f"{path}: duplicate run/process key {trace_key}")
        seen_trace_keys.add(trace_key)

        fixed_fields = (
            "run_id",
            "process_repeat",
            "configured_dpus",
            "num_tasklets",
            "actual_ranks",
            "op",
            "direction",
            "sdk_api_kind",
            "logical_distribution_class",
            "target_space",
            "transfer_bytes_per_dpu",
            "active_dpus",
            "active_ranks",
            "active_dpus_per_rank",
            "same_source_across_group",
            "phase_class",
            "source_buffer_class",
            "same_source_across_conditions",
            "source_pointer",
            "source_content_hash",
            "source_alignment_bytes",
            "target_precondition",
            "previous_op",
            "previous_direction",
        )
        fixed = {field: file_rows[0][field] for field in fixed_fields}
        file_topology: dict[str, tuple[str, ...]] = {}
        for row in file_rows:
            global_id = row["target_global_dpu_id"]
            topology = (
                row["rank_ordinal"],
                row["dpu_ordinal_in_rank"],
                row["sdk_slice_id"],
                row["sdk_member_id"],
                row["offset_bytes"],
            )
            prior = file_topology.setdefault(global_id, topology)
            if prior != topology:
                raise ValueError(f"{path}: measured DPU topology changed")
        by_sample_target: dict[
            tuple[int, int], list[dict[str, str]]
        ] = defaultdict(list)
        by_sample: dict[int, list[dict[str, str]]] = defaultdict(list)

        for row in file_rows:
            row["_trace_file"] = str(path)
            if any(row[field] != value for field, value in fixed.items()):
                raise ValueError(f"{path}: controlled fields changed")
            condition = row["condition"]
            if condition not in CONDITIONS:
                raise ValueError(f"{path}: unknown condition {condition}")
            if row["verification"] != "ok":
                raise ValueError(f"{path}: readback verification failed")
            if (
                row["op"],
                row["direction"],
                row["sdk_api_kind"],
                row["logical_distribution_class"],
            ) != (
                "dpu_copy_to",
                "TO_DPU",
                "SINGLE_COPY",
                "SHARED_REPLICATION",
            ):
                raise ValueError(f"{path}: measured transport semantics changed")
            if (row["active_dpus"], row["active_ranks"], row["active_dpus_per_rank"]) != (
                "1",
                "1",
                "1",
            ):
                raise ValueError(f"{path}: active topology changed")
            if row["phase_class"] != "CONTROLLED_PREDECESSOR_FACTORIAL":
                raise ValueError(f"{path}: phase provenance changed")
            if row["same_source_across_group"] != "1":
                raise ValueError(f"{path}: group source reuse changed")
            if row["same_source_across_conditions"] != "1":
                raise ValueError(f"{path}: condition source reuse changed")
            alignment = int(row["source_alignment_bytes"])
            if alignment != 4096 or int(row["source_pointer"], 0) % alignment != 0:
                raise ValueError(f"{path}: measured source is misaligned")
            if int(row["source_content_hash"], 0) == 0:
                raise ValueError(f"{path}: measured source hash is zero")
            if row["target_precondition"] != "ZERO_WRITTEN":
                raise ValueError(f"{path}: target precondition changed")
            if row["previous_op"] != "dpu_copy_to" or row["previous_direction"] != "TO_DPU":
                raise ValueError(f"{path}: explicit predecessor operation changed")

            start_ns = int(row["host_start_ns"])
            end_ns = int(row["host_end_ns"])
            if end_ns - start_ns != int(row["measured_ns"]):
                raise ValueError(f"{path}: measured time conservation failed")
            predecessor_start = int(row["predecessor_start_ns"])
            predecessor_end = int(row["predecessor_end_ns"])
            if predecessor_end - predecessor_start != int(row["predecessor_ns"]):
                raise ValueError(f"{path}: predecessor time conservation failed")
            precondition_start = int(row["precondition_start_ns"])
            precondition_end = int(row["precondition_end_ns"])
            if precondition_end - precondition_start != int(row["precondition_ns"]):
                raise ValueError(f"{path}: precondition time conservation failed")
            if predecessor_end > start_ns:
                raise ValueError(f"{path}: predecessor overlaps measured call")
            if int(row["ns_since_previous_sdk_event"]) != start_ns - predecessor_end:
                raise ValueError(f"{path}: predecessor gap changed")
            if int(row["source_pretouch_ns"]) <= 0:
                raise ValueError(f"{path}: source pretouch was not recorded")

            global_id = int(row["target_global_dpu_id"])
            ordinal = int(row["dpu_ordinal_in_rank"])
            if int(row["dpu_id_in_rank"]) != ordinal:
                raise ValueError(f"{path}: compatibility ordinal changed")
            expected_parity = "EVEN" if ordinal % 2 == 0 else "ODD"
            if row["ordinal_parity"] != expected_parity:
                raise ValueError(f"{path}: ordinal parity changed")
            if not (0 <= int(row["sdk_slice_id"]) < 8):
                raise ValueError(f"{path}: SDK slice ID is outside 0..7")
            if not (0 <= int(row["sdk_member_id"]) < 8):
                raise ValueError(f"{path}: SDK member ID is outside 0..7")
            topology = (
                row["rank_ordinal"],
                row["dpu_ordinal_in_rank"],
                row["sdk_slice_id"],
                row["sdk_member_id"],
                row["offset_bytes"],
            )
            prior_topology = topology_reference.setdefault(str(global_id), topology)
            if topology != prior_topology:
                raise ValueError(f"{path}: measured DPU topology changed")

            same_dpu, predecessor_bytes = expected_condition_semantics(condition)
            predecessor_id = int(row["predecessor_global_dpu_id"])
            if int(row["previous_transfer_bytes"]) != predecessor_bytes:
                raise ValueError(f"{path}: predecessor byte factor changed")
            if int(row["same_dpu_as_previous"]) != int(same_dpu):
                raise ValueError(f"{path}: predecessor locality factor changed")
            if row["same_rank_as_previous"] != "1":
                raise ValueError(f"{path}: paired predecessor left the rank")
            if int(row["predecessor_rank_ordinal"]) != int(row["rank_ordinal"]):
                raise ValueError(f"{path}: predecessor rank changed")
            predecessor_topology = file_topology.get(str(predecessor_id))
            observed_predecessor_topology = (
                row["predecessor_rank_ordinal"],
                row["predecessor_dpu_ordinal_in_rank"],
                row["predecessor_slice_id"],
                row["predecessor_member_id"],
            )
            if (
                predecessor_topology is None
                or predecessor_topology[:4] != observed_predecessor_topology
            ):
                raise ValueError(f"{path}: predecessor topology metadata changed")
            previous_offset = row["previous_offset_bytes"]
            prior_previous_offset = predecessor_offset_reference.setdefault(
                str(predecessor_id), previous_offset
            )
            if previous_offset != prior_previous_offset:
                raise ValueError(f"{path}: predecessor MRAM offset changed")
            if same_dpu:
                if predecessor_id != global_id or row["transition_class"] != "SAME_DPU":
                    raise ValueError(f"{path}: same-DPU transition changed")
            else:
                predecessor_ordinal = int(row["predecessor_dpu_ordinal_in_rank"])
                if predecessor_ordinal != (ordinal ^ 1):
                    raise ValueError(f"{path}: paired ordinal changed")
                expected_transition = "ODD_TO_EVEN" if ordinal % 2 == 0 else "EVEN_TO_ODD"
                if row["transition_class"] != expected_transition:
                    raise ValueError(f"{path}: paired transition class changed")

            sample = int(row["sample_index"])
            target = int(row["target_global_dpu_id"])
            order_index = int(row["condition_order_index"])
            expected_condition = CONDITIONS[
                (sample + target + order_index) % len(CONDITIONS)
            ]
            if condition != expected_condition:
                raise ValueError(f"{path}: condition rotation changed")
            row["_position_class"] = position_class(
                int(row["call_position_in_rank"])
            )
            by_sample_target[(sample, target)].append(row)
            by_sample[sample].append(row)
            hashes_by_size[row["transfer_bytes_per_dpu"]].add(
                row["source_content_hash"]
            )
            rows.append(row)

        configured_dpus = int(fixed["configured_dpus"])
        actual_ranks = int(fixed["actual_ranks"])
        if sorted(by_sample) != list(range(len(by_sample))):
            raise ValueError(f"{path}: sample indexes are not contiguous")
        for (sample, target), target_rows in by_sample_target.items():
            if {row["condition"] for row in target_rows} != set(CONDITIONS):
                raise ValueError(
                    f"{path}: sample {sample} target {target} condition set changed"
                )
            common = (
                "target_order_seed",
                "target_visit_position",
                "call_position_in_rank",
            )
            for field in common:
                if len({row[field] for row in target_rows}) != 1:
                    raise ValueError(f"{path}: {field} changed across conditions")
        for sample, sample_rows in by_sample.items():
            representatives: dict[int, dict[str, str]] = {}
            for row in sample_rows:
                representatives.setdefault(int(row["target_global_dpu_id"]), row)
            if sorted(representatives) != list(range(configured_dpus)):
                raise ValueError(f"{path}: sample {sample} target set changed")
            positions = sorted(
                int(row["target_visit_position"])
                for row in representatives.values()
            )
            if positions != list(range(configured_dpus)):
                raise ValueError(f"{path}: sample {sample} target order is not a permutation")
            if len({row["target_order_seed"] for row in representatives.values()}) != 1:
                raise ValueError(f"{path}: sample {sample} target seed changed")
            for rank in range(actual_ranks):
                rank_positions = sorted(
                    int(row["call_position_in_rank"])
                    for row in representatives.values()
                    if int(row["rank_ordinal"]) == rank
                )
                if rank_positions != list(range(len(rank_positions))):
                    raise ValueError(
                        f"{path}: sample {sample} rank {rank} call positions changed"
                    )

        topology_pairs: dict[int, set[tuple[str, str]]] = defaultdict(set)
        for row in file_rows:
            topology_pairs[int(row["rank_ordinal"])].add(
                (row["sdk_slice_id"], row["sdk_member_id"])
            )
        for rank, pairs in topology_pairs.items():
            if len(pairs) != configured_dpus // actual_ranks:
                raise ValueError(f"{path}: rank {rank} physical topology is not unique")

    for transfer_bytes, hashes in hashes_by_size.items():
        if len(hashes) != 1:
            raise ValueError(
                f"transfer size {transfer_bytes}: source hashes changed"
            )
    return rows


def stability_label(
    values: list[int],
    trace_files: int,
    min_samples: int,
    min_traces: int,
    spread_threshold_pct: float,
    cv_threshold_pct: float,
) -> tuple[dict[str, float | int], str]:
    stats = distribution(values)
    if len(values) < min_samples or trace_files < min_traces:
        label = "INSUFFICIENT"
    elif float(stats["spread_pct"]) <= spread_threshold_pct:
        if float(stats["cv_pct"]) <= cv_threshold_pct:
            label = "STABLE"
        else:
            label = "OUTLIER_SENSITIVE"
    else:
        label = "UNSTABLE"
    return stats, label


def summarize_conditions(
    rows: list[dict[str, str]],
    per_dpu: bool,
    min_samples: int = 20,
    min_traces: int = 5,
    spread_threshold_pct: float = 25.0,
    cv_threshold_pct: float = 25.0,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[tuple[int, str]]] = defaultdict(list)
    for row in rows:
        key_parts = [row["configured_dpus"], row["num_tasklets"]]
        if per_dpu:
            key_parts.extend(
                [
                    row["target_global_dpu_id"],
                    row["rank_ordinal"],
                    row["dpu_ordinal_in_rank"],
                    row["sdk_slice_id"],
                    row["sdk_member_id"],
                    row["ordinal_parity"],
                ]
            )
        key_parts.append(row["condition"])
        grouped[tuple(key_parts)].append(
            (int(row["measured_ns"]), row["_trace_file"])
        )

    output: list[dict[str, object]] = []
    for key, samples in sorted(grouped.items()):
        trace_count = len({sample[1] for sample in samples})
        stats, label = stability_label(
            [sample[0] for sample in samples],
            trace_count,
            min_samples,
            min_traces,
            spread_threshold_pct,
            cv_threshold_pct,
        )
        result: dict[str, object] = {
            "configured_dpus": key[0],
            "num_tasklets": key[1],
        }
        cursor = 2
        if per_dpu:
            result.update(
                {
                    "target_global_dpu_id": key[cursor],
                    "rank_ordinal": key[cursor + 1],
                    "dpu_ordinal_in_rank": key[cursor + 2],
                    "sdk_slice_id": key[cursor + 3],
                    "sdk_member_id": key[cursor + 4],
                    "ordinal_parity": key[cursor + 5],
                }
            )
            cursor += 6
        result.update(
            {
                "condition": key[cursor],
                "trace_files": trace_count,
                **stats,
                "stability": label,
            }
        )
        output.append(result)
    return output


def build_paired_samples(
    rows: list[dict[str, str]],
) -> tuple[
    dict[tuple[str, ...], dict[str, int]],
    dict[tuple[str, ...], dict[str, str]],
]:
    paired: dict[tuple[str, ...], dict[str, int]] = defaultdict(dict)
    metadata: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = (
            row["_trace_file"],
            row["process_repeat"],
            row["sample_index"],
            row["target_global_dpu_id"],
        )
        paired[key][row["condition"]] = int(row["measured_ns"])
        metadata[key] = row
    for key, values in paired.items():
        if set(values) != set(CONDITIONS):
            raise ValueError(f"incomplete factorial conditions for {key}")
    return paired, metadata


def summarize_effects(
    rows: list[dict[str, str]],
    grouping: str,
    effect_threshold_pct: float,
) -> list[dict[str, object]]:
    paired, metadata = build_paired_samples(rows)
    grouped: dict[tuple[str, ...], list[tuple[float, float]]] = defaultdict(list)

    for key, values in paired.items():
        row = metadata[key]
        for effect, lhs, rhs in COMPARISONS:
            delta = float(values[lhs] - values[rhs])
            delta_pct = 0.0 if values[rhs] == 0 else delta / values[rhs] * 100.0
            group_key = [row["configured_dpus"], row["num_tasklets"]]
            if grouping == "parity":
                group_key.append(row["ordinal_parity"])
            elif grouping == "position":
                group_key.extend([row["ordinal_parity"], row["_position_class"]])
            elif grouping == "per_dpu":
                group_key.extend(
                    [
                        row["target_global_dpu_id"],
                        row["rank_ordinal"],
                        row["dpu_ordinal_in_rank"],
                        row["sdk_slice_id"],
                        row["sdk_member_id"],
                        row["ordinal_parity"],
                    ]
                )
            group_key.extend([effect, lhs, rhs])
            grouped[tuple(group_key)].append((delta, delta_pct))

    output: list[dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        deltas = [value[0] for value in values]
        delta_pcts = [value[1] for value in values]
        p10 = quantile(deltas, 0.10)
        p90 = quantile(deltas, 0.90)
        median_pct = statistics.median(delta_pcts)
        result: dict[str, object] = {
            "configured_dpus": key[0],
            "num_tasklets": key[1],
        }
        cursor = 2
        if grouping == "parity":
            result["ordinal_parity"] = key[cursor]
            cursor += 1
        elif grouping == "position":
            result["ordinal_parity"] = key[cursor]
            result["call_position_class"] = key[cursor + 1]
            cursor += 2
        elif grouping == "per_dpu":
            result.update(
                {
                    "target_global_dpu_id": key[cursor],
                    "rank_ordinal": key[cursor + 1],
                    "dpu_ordinal_in_rank": key[cursor + 2],
                    "sdk_slice_id": key[cursor + 3],
                    "sdk_member_id": key[cursor + 4],
                    "ordinal_parity": key[cursor + 5],
                }
            )
            cursor += 6
        result.update(
            {
                "effect": key[cursor],
                "lhs_condition": key[cursor + 1],
                "rhs_condition": key[cursor + 2],
                "paired_n": len(values),
                "median_delta_ns": statistics.median(deltas),
                "p10_delta_ns": p10,
                "p90_delta_ns": p90,
                "median_delta_pct": median_pct,
                "positive_pair_pct": sum(delta > 0 for delta in deltas)
                / len(deltas)
                * 100.0,
                "effect_class": effect_class(
                    median_pct, p10, p90, effect_threshold_pct
                ),
            }
        )
        output.append(result)
    return output


def topology_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    paired_by_target = {
        row["target_global_dpu_id"]: row["predecessor_global_dpu_id"]
        for row in rows
        if row["condition"].startswith("PAIRED_DPU_")
    }
    seen: set[str] = set()
    for row in rows:
        global_id = row["target_global_dpu_id"]
        if global_id in seen:
            continue
        seen.add(global_id)
        output.append(
            {
                "target_global_dpu_id": global_id,
                "rank_ordinal": row["rank_ordinal"],
                "dpu_ordinal_in_rank": row["dpu_ordinal_in_rank"],
                "sdk_slice_id": row["sdk_slice_id"],
                "sdk_member_id": row["sdk_member_id"],
                "ordinal_parity": row["ordinal_parity"],
                "paired_global_dpu_id": paired_by_target[global_id],
            }
        )
    return sorted(output, key=lambda row: int(row["target_global_dpu_id"]))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--condition-output", type=Path, required=True)
    parser.add_argument("--per-dpu-condition-output", type=Path, required=True)
    parser.add_argument("--overall-effects-output", type=Path, required=True)
    parser.add_argument("--parity-effects-output", type=Path, required=True)
    parser.add_argument("--position-effects-output", type=Path, required=True)
    parser.add_argument("--per-dpu-effects-output", type=Path, required=True)
    parser.add_argument("--topology-output", type=Path, required=True)
    parser.add_argument("--effect-threshold-pct", type=float, default=5.0)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--min-traces", type=int, default=5)
    parser.add_argument("--spread-threshold-pct", type=float, default=25.0)
    parser.add_argument("--cv-threshold-pct", type=float, default=25.0)
    args = parser.parse_args()

    rows = read_and_validate(args.traces)
    summary_args = {
        "min_samples": args.min_samples,
        "min_traces": args.min_traces,
        "spread_threshold_pct": args.spread_threshold_pct,
        "cv_threshold_pct": args.cv_threshold_pct,
    }
    condition_summary = summarize_conditions(rows, False, **summary_args)
    per_dpu_summary = summarize_conditions(rows, True, **summary_args)
    overall_effects = summarize_effects(rows, "overall", args.effect_threshold_pct)
    parity_effects = summarize_effects(rows, "parity", args.effect_threshold_pct)
    position_effects = summarize_effects(rows, "position", args.effect_threshold_pct)
    per_dpu_effects = summarize_effects(rows, "per_dpu", args.effect_threshold_pct)
    topology = topology_rows(rows)

    write_csv(args.condition_output, condition_summary)
    write_csv(args.per_dpu_condition_output, per_dpu_summary)
    write_csv(args.overall_effects_output, overall_effects)
    write_csv(args.parity_effects_output, parity_effects)
    write_csv(args.position_effects_output, position_effects)
    write_csv(args.per_dpu_effects_output, per_dpu_effects)
    write_csv(args.topology_output, topology)

    print(f"trace_files={len(args.traces)}")
    print(f"rows={len(rows)}")
    print(f"paired_target_samples={len(rows) // len(CONDITIONS)}")
    print(f"topology_dpus={len(topology)}")
    print(
        "source_control="
        f"hash={rows[0]['source_content_hash']},"
        f"alignment={rows[0]['source_alignment_bytes']},"
        f"target_precondition={rows[0]['target_precondition']}"
    )
    print("\ncondition summaries:")
    for row in condition_summary:
        print(
            f"  condition={row['condition']} n={row['n']} "
            f"median_ns={row['median_ns']:.1f} p10_ns={row['p10_ns']:.1f} "
            f"p90_ns={row['p90_ns']:.1f} cv_pct={row['cv_pct']:.2f} "
            f"spread_pct={row['spread_pct']:.2f} stability={row['stability']}"
        )
    print("\noverall factorial effects:")
    for row in overall_effects:
        print(
            f"  effect={row['effect']} n={row['paired_n']} "
            f"median_delta_ns={row['median_delta_ns']:.1f} "
            f"p10_delta_ns={row['p10_delta_ns']:.1f} "
            f"p90_delta_ns={row['p90_delta_ns']:.1f} "
            f"median_delta_pct={row['median_delta_pct']:.2f} "
            f"class={row['effect_class']}"
        )
    print("\nparity factorial effects:")
    for row in parity_effects:
        print(
            f"  parity={row['ordinal_parity']} effect={row['effect']} "
            f"n={row['paired_n']} median_delta_pct={row['median_delta_pct']:.2f} "
            f"p10_delta_ns={row['p10_delta_ns']:.1f} "
            f"p90_delta_ns={row['p90_delta_ns']:.1f} "
            f"class={row['effect_class']}"
        )
    status_counts: dict[str, int] = defaultdict(int)
    for row in per_dpu_summary:
        status_counts[str(row["stability"])] += 1
    print(
        "\nper_dpu_stability="
        + ",".join(f"{key}:{value}" for key, value in sorted(status_counts.items()))
    )


if __name__ == "__main__":
    main()
