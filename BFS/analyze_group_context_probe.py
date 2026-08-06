#!/usr/bin/env python3
"""Validate and analyze the full-DPU BFS transfer-group context probe."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


CONDITIONS = (
    "H2D_GROUP_THEN_H2D_GROUP",
    "D2H_GROUP_THEN_H2D_GROUP",
    "LAUNCH_D2H_GROUP_THEN_H2D_GROUP",
)
CONDITION_INDEX = {condition: index for index, condition in enumerate(CONDITIONS)}
COMPARISONS = (
    (
        "D2H_GROUP_HISTORY_EFFECT",
        "D2H_GROUP_THEN_H2D_GROUP",
        "H2D_GROUP_THEN_H2D_GROUP",
    ),
    (
        "AFTER_LAUNCH_GROUP_EFFECT",
        "LAUNCH_D2H_GROUP_THEN_H2D_GROUP",
        "D2H_GROUP_THEN_H2D_GROUP",
    ),
    (
        "COMBINED_ITERATIVE_GROUP_EFFECT",
        "LAUNCH_D2H_GROUP_THEN_H2D_GROUP",
        "H2D_GROUP_THEN_H2D_GROUP",
    ),
)
REQUIRED_FIELDS = {
    "run_id",
    "process_repeat",
    "configured_dpus",
    "num_tasklets",
    "actual_ranks",
    "sample_index",
    "order_index",
    "condition",
    "predecessor_chain",
    "group_index",
    "group_size",
    "target_global_dpu_id",
    "rank_ordinal",
    "dpu_id_in_rank",
    "rank_boundary_before",
    "same_rank_as_previous",
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
    "group_previous_op",
    "group_previous_direction",
    "direction_switched",
    "after_launch",
    "predecessor_group_ns",
    "launch_ns",
    "ns_since_previous_sdk_event",
    "source_pretouch_ns",
    "group_start_ns",
    "group_end_ns",
    "host_start_ns",
    "host_end_ns",
    "measured_ns",
    "measured_group_ns",
    "verification",
}


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: Iterable[float]) -> dict[str, float | int]:
    data = list(values)
    median = statistics.median(data)
    mean = statistics.mean(data)
    stdev = statistics.stdev(data) if len(data) > 1 else 0.0
    p10 = quantile(data, 0.10)
    p90 = quantile(data, 0.90)
    return {
        "n": len(data),
        "median_ns": median,
        "p10_ns": p10,
        "p90_ns": p90,
        "mean_ns": mean,
        "cv_pct": 0.0 if mean == 0 else stdev / mean * 100.0,
        "spread_pct": 0.0 if median == 0 else (p90 - p10) / median * 100.0,
    }


def effect_class(
    median_pct: float, p10_delta: float, p90_delta: float, threshold_pct: float
) -> str:
    if median_pct >= threshold_pct and p10_delta > 0:
        return "CONSISTENT_SLOWER"
    if median_pct <= -threshold_pct and p90_delta < 0:
        return "CONSISTENT_FASTER"
    return "MIXED_OR_SMALL"


def _expected_predecessor(condition: str) -> tuple[str, str, str, str]:
    if condition == "H2D_GROUP_THEN_H2D_GROUP":
        return ("dpu_copy_to", "TO_DPU", "0", "0")
    if condition == "D2H_GROUP_THEN_H2D_GROUP":
        return ("dpu_copy_from", "FROM_DPU", "1", "0")
    return ("dpu_copy_from", "FROM_DPU", "1", "1")


def read_and_validate(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_trace_keys: set[tuple[str, str]] = set()
    source_hashes_by_size: dict[str, set[str]] = defaultdict(set)
    topology_reference: dict[str, tuple[str, str, str]] = {}

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

        file_fixed_fields = (
            "run_id",
            "process_repeat",
            "configured_dpus",
            "num_tasklets",
            "actual_ranks",
            "group_size",
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
        )
        fixed = {field: file_rows[0][field] for field in file_fixed_fields}
        by_sample_condition: dict[tuple[int, str], list[dict[str, str]]] = (
            defaultdict(list)
        )
        by_sample: dict[int, set[str]] = defaultdict(set)

        for row in file_rows:
            row["_trace_file"] = str(path)
            if any(row[field] != value for field, value in fixed.items()):
                raise ValueError(f"{path}: controlled group fields changed")
            condition = row["condition"]
            if condition not in CONDITION_INDEX:
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
                raise ValueError(f"{path}: single-copy active topology changed")
            if row["same_source_across_group"] != "1":
                raise ValueError(f"{path}: group source reuse invariant is false")
            if row["same_source_across_conditions"] != "1":
                raise ValueError(f"{path}: condition source reuse invariant is false")
            alignment = int(row["source_alignment_bytes"])
            if alignment != 4096 or int(row["source_pointer"], 0) % alignment != 0:
                raise ValueError(f"{path}: fixed source pointer is misaligned")
            if int(row["source_content_hash"], 0) == 0:
                raise ValueError(f"{path}: fixed source hash is zero")
            if row["target_precondition"] != "ZERO_WRITTEN":
                raise ValueError(f"{path}: target precondition changed")

            sample = int(row["sample_index"])
            order = int(row["order_index"])
            expected_condition = CONDITIONS[(sample + order) % len(CONDITIONS)]
            if condition != expected_condition:
                raise ValueError(f"{path}: Latin-order invariant failed")
            actual_predecessor = (
                row["group_previous_op"],
                row["group_previous_direction"],
                row["direction_switched"],
                row["after_launch"],
            )
            if actual_predecessor != _expected_predecessor(condition):
                raise ValueError(f"{path}: predecessor semantics changed")
            if condition.startswith("LAUNCH_"):
                if int(row["launch_ns"]) <= 0:
                    raise ValueError(f"{path}: launch condition has zero launch_ns")
            elif int(row["launch_ns"]) != 0:
                raise ValueError(f"{path}: non-launch condition has launch_ns")

            start_ns = int(row["host_start_ns"])
            end_ns = int(row["host_end_ns"])
            if end_ns - start_ns != int(row["measured_ns"]):
                raise ValueError(f"{path}: call time conservation failed")
            group_start = int(row["group_start_ns"])
            group_end = int(row["group_end_ns"])
            if group_end - group_start != int(row["measured_group_ns"]):
                raise ValueError(f"{path}: group time conservation failed")
            if start_ns < group_start or end_ns > group_end:
                raise ValueError(f"{path}: call lies outside measured group")

            global_id = row["target_global_dpu_id"]
            topology = (
                row["rank_ordinal"],
                row["dpu_id_in_rank"],
                row["offset_bytes"],
            )
            prior_topology = topology_reference.setdefault(global_id, topology)
            if topology != prior_topology:
                raise ValueError(f"{path}: DPU topology or target offset changed")

            by_sample_condition[(sample, condition)].append(row)
            by_sample[sample].add(condition)
            source_hashes_by_size[row["transfer_bytes_per_dpu"]].add(
                row["source_content_hash"]
            )
            rows.append(row)

        expected_samples = list(range(len(by_sample)))
        if sorted(by_sample) != expected_samples:
            raise ValueError(f"{path}: sample indexes are not contiguous from zero")
        for sample, observed in by_sample.items():
            if observed != set(CONDITIONS):
                raise ValueError(f"{path}: sample {sample} has conditions {observed}")

        configured_dpus = int(fixed["configured_dpus"])
        for (sample, condition), group_rows in by_sample_condition.items():
            ordered = sorted(group_rows, key=lambda row: int(row["group_index"]))
            indexes = [int(row["group_index"]) for row in ordered]
            global_ids = [int(row["target_global_dpu_id"]) for row in ordered]
            if indexes != list(range(configured_dpus)):
                raise ValueError(
                    f"{path}: sample {sample} {condition} group indexes changed"
                )
            if global_ids != list(range(configured_dpus)):
                raise ValueError(
                    f"{path}: sample {sample} {condition} DPU order changed"
                )
            common_fields = (
                "group_start_ns",
                "group_end_ns",
                "measured_group_ns",
                "predecessor_group_ns",
                "launch_ns",
                "source_pretouch_ns",
            )
            for field in common_fields:
                if len({row[field] for row in ordered}) != 1:
                    raise ValueError(f"{path}: {field} changed within group")
            for index, row in enumerate(ordered):
                expected_boundary = index == 0 or (
                    ordered[index - 1]["rank_ordinal"] != row["rank_ordinal"]
                )
                if int(row["rank_boundary_before"]) != int(expected_boundary):
                    raise ValueError(f"{path}: rank boundary label changed")
                if int(row["same_rank_as_previous"]) != int(
                    index > 0 and not expected_boundary
                ):
                    raise ValueError(f"{path}: previous-rank label changed")
                if index > 0 and int(row["host_start_ns"]) < int(
                    ordered[index - 1]["host_end_ns"]
                ):
                    raise ValueError(f"{path}: measured calls overlap or reorder")

    for transfer_bytes, hashes in source_hashes_by_size.items():
        if len(hashes) != 1:
            raise ValueError(
                f"transfer size {transfer_bytes}: controlled source hashes changed"
            )
    return rows


def _stability(
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
    elif (
        float(stats["spread_pct"]) <= spread_threshold_pct
        and float(stats["cv_pct"]) <= cv_threshold_pct
    ):
        label = "STABLE"
    else:
        label = "UNSTABLE"
    return stats, label


def summarize_per_dpu_conditions(
    rows: list[dict[str, str]],
    min_samples: int = 20,
    min_traces: int = 5,
    spread_threshold_pct: float = 25.0,
    cv_threshold_pct: float = 25.0,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[tuple[int, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["configured_dpus"],
            row["num_tasklets"],
            row["target_global_dpu_id"],
            row["rank_ordinal"],
            row["dpu_id_in_rank"],
            row["group_index"],
            row["transfer_bytes_per_dpu"],
            row["offset_bytes"],
            row["condition"],
        )
        grouped[key].append((int(row["measured_ns"]), row["_trace_file"]))

    output: list[dict[str, object]] = []
    for key, samples in sorted(grouped.items()):
        trace_files = len({sample[1] for sample in samples})
        stats, label = _stability(
            [sample[0] for sample in samples],
            trace_files,
            min_samples,
            min_traces,
            spread_threshold_pct,
            cv_threshold_pct,
        )
        output.append(
            {
                "configured_dpus": key[0],
                "num_tasklets": key[1],
                "target_global_dpu_id": key[2],
                "rank_ordinal": key[3],
                "dpu_id_in_rank": key[4],
                "group_index": key[5],
                "transfer_bytes_per_dpu": key[6],
                "offset_bytes": key[7],
                "condition": key[8],
                "trace_files": trace_files,
                **stats,
                "stability": label,
            }
        )
    return output


def group_measurements(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    measurements: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = (
            row["_trace_file"],
            row["process_repeat"],
            row["sample_index"],
            row["condition"],
        )
        if key in seen:
            continue
        seen.add(key)
        measurements.append(row)
    return measurements


def summarize_group_conditions(
    rows: list[dict[str, str]],
    min_samples: int = 20,
    min_traces: int = 5,
    spread_threshold_pct: float = 25.0,
    cv_threshold_pct: float = 25.0,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[tuple[int, str]]] = defaultdict(list)
    for row in group_measurements(rows):
        key = (
            row["configured_dpus"],
            row["num_tasklets"],
            row["actual_ranks"],
            row["group_size"],
            row["transfer_bytes_per_dpu"],
            row["condition"],
        )
        grouped[key].append((int(row["measured_group_ns"]), row["_trace_file"]))

    output: list[dict[str, object]] = []
    for key, samples in sorted(grouped.items()):
        trace_files = len({sample[1] for sample in samples})
        stats, label = _stability(
            [sample[0] for sample in samples],
            trace_files,
            min_samples,
            min_traces,
            spread_threshold_pct,
            cv_threshold_pct,
        )
        output.append(
            {
                "configured_dpus": key[0],
                "num_tasklets": key[1],
                "actual_ranks": key[2],
                "group_size": key[3],
                "transfer_bytes_per_dpu": key[4],
                "condition": key[5],
                "trace_files": trace_files,
                **stats,
                "stability": label,
            }
        )
    return output


def _summarize_pairs(
    measurements: list[dict[str, str]],
    value_field: str,
    per_dpu: bool,
    effect_threshold_pct: float,
) -> list[dict[str, object]]:
    paired: dict[tuple[str, ...], dict[str, int]] = defaultdict(dict)
    metadata: dict[tuple[str, ...], dict[str, str]] = {}
    for row in measurements:
        key_parts = [
            row["_trace_file"],
            row["process_repeat"],
            row["sample_index"],
        ]
        if per_dpu:
            key_parts.append(row["target_global_dpu_id"])
        key = tuple(key_parts)
        paired[key][row["condition"]] = int(row[value_field])
        metadata[key] = row

    grouped: dict[tuple[str, ...], list[tuple[float, float]]] = defaultdict(list)
    for key, values in paired.items():
        if set(values) != set(CONDITIONS):
            raise ValueError(f"incomplete paired conditions for {key}")
        row = metadata[key]
        for effect, lhs, rhs in COMPARISONS:
            delta = float(values[lhs] - values[rhs])
            delta_pct = 0.0 if values[rhs] == 0 else delta / values[rhs] * 100.0
            group_key = [
                row["configured_dpus"],
                row["num_tasklets"],
            ]
            if per_dpu:
                group_key.extend(
                    [
                        row["target_global_dpu_id"],
                        row["rank_ordinal"],
                        row["dpu_id_in_rank"],
                        row["group_index"],
                    ]
                )
            else:
                group_key.extend([row["actual_ranks"], row["group_size"]])
            group_key.extend([effect, lhs, rhs])
            grouped[tuple(group_key)].append((delta, delta_pct))

    output: list[dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        deltas = [value[0] for value in values]
        delta_pcts = [value[1] for value in values]
        p10 = quantile(deltas, 0.10)
        p90 = quantile(deltas, 0.90)
        median_pct = statistics.median(delta_pcts)
        row: dict[str, object] = {
            "configured_dpus": key[0],
            "num_tasklets": key[1],
        }
        cursor = 2
        if per_dpu:
            row.update(
                {
                    "target_global_dpu_id": key[cursor],
                    "rank_ordinal": key[cursor + 1],
                    "dpu_id_in_rank": key[cursor + 2],
                    "group_index": key[cursor + 3],
                }
            )
            cursor += 4
        else:
            row.update(
                {
                    "actual_ranks": key[cursor],
                    "group_size": key[cursor + 1],
                }
            )
            cursor += 2
        row.update(
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
        output.append(row)
    return output


def summarize_per_dpu_pairs(
    rows: list[dict[str, str]], effect_threshold_pct: float
) -> list[dict[str, object]]:
    return _summarize_pairs(rows, "measured_ns", True, effect_threshold_pct)


def summarize_group_pairs(
    rows: list[dict[str, str]], effect_threshold_pct: float
) -> list[dict[str, object]]:
    return _summarize_pairs(
        group_measurements(rows), "measured_group_ns", False, effect_threshold_pct
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty analysis table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--per-dpu-summary-output", type=Path, required=True)
    parser.add_argument("--per-dpu-paired-output", type=Path, required=True)
    parser.add_argument("--group-summary-output", type=Path, required=True)
    parser.add_argument("--group-paired-output", type=Path, required=True)
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
    per_dpu_summaries = summarize_per_dpu_conditions(rows, **summary_args)
    group_summaries = summarize_group_conditions(rows, **summary_args)
    per_dpu_pairs = summarize_per_dpu_pairs(rows, args.effect_threshold_pct)
    group_pairs = summarize_group_pairs(rows, args.effect_threshold_pct)
    write_csv(args.per_dpu_summary_output, per_dpu_summaries)
    write_csv(args.per_dpu_paired_output, per_dpu_pairs)
    write_csv(args.group_summary_output, group_summaries)
    write_csv(args.group_paired_output, group_pairs)

    measured_groups = group_measurements(rows)
    print(f"trace_files={len(args.traces)}")
    print(f"per_dpu_rows={len(rows)}")
    print(f"measured_groups={len(measured_groups)}")
    print(f"group_size={rows[0]['group_size']}")
    print(
        "source_control="
        f"shared_group_pointer,content_hash={rows[0]['source_content_hash']},"
        f"alignment={rows[0]['source_alignment_bytes']},"
        f"target_precondition={rows[0]['target_precondition']}"
    )
    print("\ngroup condition summaries:")
    for row in group_summaries:
        print(
            f"  condition={row['condition']} n={row['n']} "
            f"traces={row['trace_files']} median_ns={row['median_ns']:.1f} "
            f"p10_ns={row['p10_ns']:.1f} p90_ns={row['p90_ns']:.1f} "
            f"cv_pct={row['cv_pct']:.2f} spread_pct={row['spread_pct']:.2f} "
            f"stability={row['stability']}"
        )
    print("\ngroup paired effects:")
    for row in group_pairs:
        print(
            f"  effect={row['effect']} n={row['paired_n']} "
            f"median_delta_ns={row['median_delta_ns']:.1f} "
            f"p10_delta_ns={row['p10_delta_ns']:.1f} "
            f"p90_delta_ns={row['p90_delta_ns']:.1f} "
            f"median_delta_pct={row['median_delta_pct']:.2f} "
            f"class={row['effect_class']}"
        )
    stable_per_dpu = sum(row["stability"] == "STABLE" for row in per_dpu_summaries)
    print(
        "\nper_dpu_stability="
        f"{stable_per_dpu}/{len(per_dpu_summaries)} stable condition keys"
    )


if __name__ == "__main__":
    main()
