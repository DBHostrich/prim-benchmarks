#!/usr/bin/env python3
"""Validate and analyze the controlled BFS API-order probe."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_group_context_probe import distribution, effect_class, quantile


CONDITIONS = (
    "CONTIGUOUS_FRONTIER_GROUP",
    "VISITED_FRONTIER_PARAMS_PER_DPU",
    "FRONTIER_PARAMS_PER_DPU",
    "D2H_MERGE_FRONTIER_PARAMS_PER_DPU",
)
COMPARISONS = (
    (
        "PARAMS_INTERLEAVING_EFFECT",
        "FRONTIER_PARAMS_PER_DPU",
        "CONTIGUOUS_FRONTIER_GROUP",
    ),
    (
        "VISITED_PREDECESSOR_EFFECT",
        "VISITED_FRONTIER_PARAMS_PER_DPU",
        "FRONTIER_PARAMS_PER_DPU",
    ),
    (
        "D2H_GROUP_HISTORY_EFFECT",
        "D2H_MERGE_FRONTIER_PARAMS_PER_DPU",
        "FRONTIER_PARAMS_PER_DPU",
    ),
    (
        "ITERATIVE_VS_INIT_ORDER_EFFECT",
        "D2H_MERGE_FRONTIER_PARAMS_PER_DPU",
        "VISITED_FRONTIER_PARAMS_PER_DPU",
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
    "api_order_class",
    "group_index",
    "group_size",
    "target_global_dpu_id",
    "rank_ordinal",
    "dpu_id_in_rank",
    "rank_boundary_before",
    "same_rank_as_previous_sdk_event",
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
    "previous_event_role",
    "previous_direction",
    "previous_transfer_bytes",
    "previous_target_global_dpu_id",
    "same_dpu_as_previous",
    "direction_switched",
    "params_transfer_bytes",
    "interleaved_params",
    "predecessor_d2h_group_ns",
    "source_pretouch_ns",
    "ns_since_previous_sdk_event",
    "sequence_start_ns",
    "sequence_end_ns",
    "sequence_span_ns",
    "host_start_ns",
    "host_end_ns",
    "measured_ns",
    "frontier_sum_ns",
    "verification",
}


def _expected_previous(
    condition: str, index: int, group_size: int, transfer_bytes: int
) -> tuple[str, str, str, int, int, int, int]:
    if condition == "VISITED_FRONTIER_PARAMS_PER_DPU":
        return (
            "dpu_copy_to",
            "visited_control",
            "TO_DPU",
            transfer_bytes,
            index,
            1,
            0,
        )
    if index == 0:
        if condition == "D2H_MERGE_FRONTIER_PARAMS_PER_DPU":
            return (
                "dpu_copy_from",
                "frontier_readback",
                "FROM_DPU",
                transfer_bytes,
                group_size - 1,
                0,
                1,
            )
        return (
            "dpu_copy_to",
            "frontier_precondition",
            "TO_DPU",
            transfer_bytes,
            group_size - 1,
            0,
            0,
        )
    if condition == "CONTIGUOUS_FRONTIER_GROUP":
        return (
            "dpu_copy_to",
            "measured_frontier",
            "TO_DPU",
            transfer_bytes,
            index - 1,
            0,
            0,
        )
    return (
        "dpu_copy_to",
        "params_control",
        "TO_DPU",
        48,
        index - 1,
        0,
        0,
    )


def read_and_validate(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_trace_keys: set[tuple[str, str]] = set()
    hashes_by_size: dict[str, set[str]] = defaultdict(set)
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

        fixed_fields = (
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
            "params_transfer_bytes",
        )
        fixed = {field: file_rows[0][field] for field in fixed_fields}
        by_group: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
        by_sample: dict[int, set[str]] = defaultdict(set)

        for row in file_rows:
            row["_trace_file"] = str(path)
            if any(row[field] != value for field, value in fixed.items()):
                raise ValueError(f"{path}: controlled fields changed")
            if row["condition"] not in CONDITIONS:
                raise ValueError(f"{path}: unknown condition {row['condition']}")
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
            if row["same_source_across_group"] != "1":
                raise ValueError(f"{path}: group source reuse invariant changed")
            if row["same_source_across_conditions"] != "1":
                raise ValueError(f"{path}: condition source reuse invariant changed")
            alignment = int(row["source_alignment_bytes"])
            if alignment != 4096 or int(row["source_pointer"], 0) % alignment != 0:
                raise ValueError(f"{path}: fixed source pointer is misaligned")
            if int(row["source_content_hash"], 0) == 0:
                raise ValueError(f"{path}: fixed source hash is zero")
            if row["target_precondition"] != "ZERO_WRITTEN":
                raise ValueError(f"{path}: target precondition changed")
            if int(row["params_transfer_bytes"]) != 48:
                raise ValueError(f"{path}: params physical transfer is not 48 B")

            sample = int(row["sample_index"])
            order = int(row["order_index"])
            expected_condition = CONDITIONS[(sample + order) % len(CONDITIONS)]
            if row["condition"] != expected_condition:
                raise ValueError(f"{path}: Latin-order invariant failed")
            expected_order_class = {
                "CONTIGUOUS_FRONTIER_GROUP": "CONTIGUOUS_CONTROL",
                "VISITED_FRONTIER_PARAMS_PER_DPU": "INIT_LIKE_ORDER",
                "FRONTIER_PARAMS_PER_DPU": "PARAMS_INTERLEAVED_CONTROL",
                "D2H_MERGE_FRONTIER_PARAMS_PER_DPU": "ITERATIVE_LIKE_ORDER",
            }[row["condition"]]
            if row["api_order_class"] != expected_order_class:
                raise ValueError(f"{path}: API-order class changed")
            if row["phase_class"] != "CONTROLLED_API_ORDER_PROBE":
                raise ValueError(f"{path}: probe phase provenance changed")
            if int(row["source_pretouch_ns"]) <= 0:
                raise ValueError(f"{path}: source pretouch was not recorded")
            start_ns = int(row["host_start_ns"])
            end_ns = int(row["host_end_ns"])
            if end_ns - start_ns != int(row["measured_ns"]):
                raise ValueError(f"{path}: call time conservation failed")
            sequence_start = int(row["sequence_start_ns"])
            sequence_end = int(row["sequence_end_ns"])
            if sequence_end - sequence_start != int(row["sequence_span_ns"]):
                raise ValueError(f"{path}: sequence time conservation failed")
            if start_ns < sequence_start or end_ns > sequence_end:
                raise ValueError(f"{path}: call lies outside sequence")

            global_id = row["target_global_dpu_id"]
            topology = (
                row["rank_ordinal"],
                row["dpu_id_in_rank"],
                row["offset_bytes"],
            )
            previous_topology = topology_reference.setdefault(global_id, topology)
            if topology != previous_topology:
                raise ValueError(f"{path}: topology or frontier offset changed")

            by_group[(sample, row["condition"])].append(row)
            by_sample[sample].add(row["condition"])
            hashes_by_size[row["transfer_bytes_per_dpu"]].add(
                row["source_content_hash"]
            )
            rows.append(row)

        if sorted(by_sample) != list(range(len(by_sample))):
            raise ValueError(f"{path}: sample indexes are not contiguous")
        for sample, conditions in by_sample.items():
            if conditions != set(CONDITIONS):
                raise ValueError(f"{path}: sample {sample} condition set changed")

        group_size = int(fixed["group_size"])
        transfer_bytes = int(fixed["transfer_bytes_per_dpu"])
        for (sample, condition), group_rows in by_group.items():
            ordered = sorted(group_rows, key=lambda row: int(row["group_index"]))
            if [int(row["group_index"]) for row in ordered] != list(
                range(group_size)
            ):
                raise ValueError(f"{path}: group indexes changed")
            if [int(row["target_global_dpu_id"]) for row in ordered] != list(
                range(group_size)
            ):
                raise ValueError(f"{path}: measured DPU order changed")
            common_fields = (
                "sequence_start_ns",
                "sequence_end_ns",
                "sequence_span_ns",
                "frontier_sum_ns",
                "predecessor_d2h_group_ns",
                "source_pretouch_ns",
            )
            for field in common_fields:
                if len({row[field] for row in ordered}) != 1:
                    raise ValueError(f"{path}: {field} changed within sequence")
            if sum(int(row["measured_ns"]) for row in ordered) != int(
                ordered[0]["frontier_sum_ns"]
            ):
                raise ValueError(f"{path}: frontier sum conservation failed")

            expected_interleaved = condition != "CONTIGUOUS_FRONTIER_GROUP"
            expected_d2h = condition == "D2H_MERGE_FRONTIER_PARAMS_PER_DPU"
            for index, row in enumerate(ordered):
                expected_previous = _expected_previous(
                    condition, index, group_size, transfer_bytes
                )
                actual_previous = (
                    row["previous_op"],
                    row["previous_event_role"],
                    row["previous_direction"],
                    int(row["previous_transfer_bytes"]),
                    int(row["previous_target_global_dpu_id"]),
                    int(row["same_dpu_as_previous"]),
                    int(row["direction_switched"]),
                )
                if actual_previous != expected_previous:
                    raise ValueError(
                        f"{path}: previous-event semantics changed for "
                        f"{condition} DPU {index}"
                    )
                if int(row["interleaved_params"]) != int(expected_interleaved):
                    raise ValueError(f"{path}: interleaved params label changed")
                if expected_d2h != (int(row["predecessor_d2h_group_ns"]) > 0):
                    raise ValueError(f"{path}: D2H predecessor label changed")
                previous_target = int(row["previous_target_global_dpu_id"])
                expected_same_rank = (
                    ordered[previous_target]["rank_ordinal"] == row["rank_ordinal"]
                )
                if int(row["same_rank_as_previous_sdk_event"]) != int(
                    expected_same_rank
                ):
                    raise ValueError(f"{path}: previous-rank label changed")
                expected_boundary = index == 0 or (
                    ordered[index - 1]["rank_ordinal"] != row["rank_ordinal"]
                )
                if int(row["rank_boundary_before"]) != int(expected_boundary):
                    raise ValueError(f"{path}: rank boundary label changed")
                if int(row["ns_since_previous_sdk_event"]) < 0:
                    raise ValueError(f"{path}: negative SDK-event gap")

    for transfer_bytes, hashes in hashes_by_size.items():
        if len(hashes) != 1:
            raise ValueError(
                f"transfer size {transfer_bytes}: source hashes changed"
            )
    return rows


def group_measurements(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = (
            row["_trace_file"],
            row["process_repeat"],
            row["sample_index"],
            row["condition"],
        )
        if key not in seen:
            seen.add(key)
            output.append(row)
    return output


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


def summarize_conditions(
    rows: list[dict[str, str]],
    per_dpu: bool,
    min_samples: int = 20,
    min_traces: int = 5,
    spread_threshold_pct: float = 25.0,
    cv_threshold_pct: float = 25.0,
) -> list[dict[str, object]]:
    measurements = rows if per_dpu else group_measurements(rows)
    value_field = "measured_ns" if per_dpu else "frontier_sum_ns"
    grouped: dict[tuple[str, ...], list[tuple[int, str]]] = defaultdict(list)
    for row in measurements:
        key_parts = [row["configured_dpus"], row["num_tasklets"]]
        if per_dpu:
            key_parts.extend(
                [
                    row["target_global_dpu_id"],
                    row["rank_ordinal"],
                    row["dpu_id_in_rank"],
                    row["group_index"],
                ]
            )
        else:
            key_parts.extend([row["actual_ranks"], row["group_size"]])
        key_parts.extend([row["transfer_bytes_per_dpu"], row["condition"]])
        grouped[tuple(key_parts)].append(
            (int(row[value_field]), row["_trace_file"])
        )

    output: list[dict[str, object]] = []
    for key, samples in sorted(grouped.items()):
        trace_count = len({sample[1] for sample in samples})
        stats, label = _stability(
            [sample[0] for sample in samples],
            trace_count,
            min_samples,
            min_traces,
            spread_threshold_pct,
            cv_threshold_pct,
        )
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
                "transfer_bytes_per_dpu": key[cursor],
                "condition": key[cursor + 1],
                "trace_files": trace_count,
                **stats,
                "stability": label,
            }
        )
        output.append(row)
    return output


def summarize_pairs(
    rows: list[dict[str, str]], per_dpu: bool, effect_threshold_pct: float
) -> list[dict[str, object]]:
    measurements = rows if per_dpu else group_measurements(rows)
    value_field = "measured_ns" if per_dpu else "frontier_sum_ns"
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
            group_key = [row["configured_dpus"], row["num_tasklets"]]
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
    per_dpu_summaries = summarize_conditions(rows, True, **summary_args)
    group_summaries = summarize_conditions(rows, False, **summary_args)
    per_dpu_pairs = summarize_pairs(rows, True, args.effect_threshold_pct)
    group_pairs = summarize_pairs(rows, False, args.effect_threshold_pct)
    write_csv(args.per_dpu_summary_output, per_dpu_summaries)
    write_csv(args.per_dpu_paired_output, per_dpu_pairs)
    write_csv(args.group_summary_output, group_summaries)
    write_csv(args.group_paired_output, group_pairs)

    print(f"trace_files={len(args.traces)}")
    print(f"per_dpu_rows={len(rows)}")
    print(f"measured_sequences={len(group_measurements(rows))}")
    print(f"group_size={rows[0]['group_size']}")
    print(
        "source_control="
        f"content_hash={rows[0]['source_content_hash']},"
        f"alignment={rows[0]['source_alignment_bytes']},"
        f"frontier_target_precondition={rows[0]['target_precondition']}"
    )
    print("\ngroup frontier-sum summaries:")
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
