#!/usr/bin/env python3
"""Validate and analyze the controlled BFS transfer-context probe."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


CONDITIONS = (
    "COPY_TO_THEN_COPY_TO",
    "COPY_FROM_THEN_COPY_TO",
    "LAUNCH_COPY_FROM_THEN_COPY_TO",
)
CONDITION_INDEX = {condition: index for index, condition in enumerate(CONDITIONS)}
FIXED_FIELDS = (
    "configured_dpus",
    "num_tasklets",
    "actual_ranks",
    "target_global_dpu_id",
    "rank_ordinal",
    "dpu_id_in_rank",
    "op",
    "direction",
    "sdk_api_kind",
    "target_space",
    "transfer_bytes_per_dpu",
    "offset_bytes",
    "source_buffer_class",
    "same_source_across_conditions",
    "source_pointer",
    "source_content_hash",
    "source_alignment_bytes",
    "target_precondition",
)
REQUIRED_FIELDS = set(
    FIXED_FIELDS
    + (
        "run_id",
        "process_repeat",
        "sample_index",
        "order_index",
        "condition",
        "predecessor_chain",
        "previous_op",
        "previous_direction",
        "direction_switched",
        "after_launch",
        "predecessor_ns",
        "launch_ns",
        "ns_since_previous_sdk_event",
        "source_pretouch_ns",
        "host_start_ns",
        "host_end_ns",
        "measured_ns",
        "verification",
    )
)


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
    # statistics.fmean was added in Python 3.8.  Some UPMEM hosts still use
    # Python 3.7, where statistics.mean provides the same semantics here.
    mean = statistics.mean(data)
    stdev = statistics.stdev(data) if len(data) > 1 else 0.0
    return {
        "n": len(data),
        "median_ns": median,
        "p10_ns": quantile(data, 0.10),
        "p90_ns": quantile(data, 0.90),
        "mean_ns": mean,
        "cv_pct": 0.0 if mean == 0 else stdev / mean * 100.0,
        "spread_pct": (
            0.0
            if median == 0
            else (quantile(data, 0.90) - quantile(data, 0.10)) / median * 100.0
        ),
    }


def read_and_validate(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_trace_keys: set[tuple[str, str, str]] = set()
    source_hashes_by_size: dict[str, set[str]] = defaultdict(set)

    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            missing = REQUIRED_FIELDS - fields
            if missing:
                raise ValueError(f"{path}: missing fields {sorted(missing)}")
            file_rows = list(reader)
        if not file_rows:
            raise ValueError(f"{path}: empty trace")

        fixed = {field: file_rows[0][field] for field in FIXED_FIELDS}
        trace_key = (
            file_rows[0]["run_id"],
            file_rows[0]["process_repeat"],
            file_rows[0]["target_global_dpu_id"],
        )
        if trace_key in seen_trace_keys:
            raise ValueError(f"{path}: duplicate run/process/target key {trace_key}")
        seen_trace_keys.add(trace_key)

        by_sample: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in file_rows:
            row["_trace_file"] = str(path)
            if any(row[field] != value for field, value in fixed.items()):
                raise ValueError(f"{path}: controlled transfer fields changed")
            if row["verification"] != "ok":
                raise ValueError(f"{path}: readback verification failed")
            condition = row["condition"]
            if condition not in CONDITION_INDEX:
                raise ValueError(f"{path}: unknown condition {condition}")
            start_ns = int(row["host_start_ns"])
            end_ns = int(row["host_end_ns"])
            measured_ns = int(row["measured_ns"])
            if end_ns < start_ns or end_ns - start_ns != measured_ns:
                raise ValueError(f"{path}: measured_ns violates time conservation")
            if row["op"] != "dpu_copy_to" or row["direction"] != "TO_DPU":
                raise ValueError(f"{path}: measured operation changed")
            if row["source_buffer_class"] != "SHARED_FIXED_BUFFER":
                raise ValueError(f"{path}: final source buffer is not fixed")
            if row["same_source_across_conditions"] != "1":
                raise ValueError(f"{path}: source reuse invariant is false")
            if int(row["source_pointer"], 0) == 0:
                raise ValueError(f"{path}: fixed source pointer is null")
            if int(row["source_content_hash"], 0) == 0:
                raise ValueError(f"{path}: fixed source hash is zero")
            source_alignment = int(row["source_alignment_bytes"])
            if source_alignment != 4096:
                raise ValueError(f"{path}: source alignment is {source_alignment}")
            if int(row["source_pointer"], 0) % source_alignment != 0:
                raise ValueError(f"{path}: fixed source pointer is misaligned")
            if row["target_precondition"] != "ZERO_WRITTEN":
                raise ValueError(f"{path}: target precondition changed")
            if int(row["source_pretouch_ns"]) <= 0:
                raise ValueError(f"{path}: source buffer pretouch was not recorded")

            sample_index = int(row["sample_index"])
            order_index = int(row["order_index"])
            expected_condition = CONDITIONS[(sample_index + order_index) % len(CONDITIONS)]
            if condition != expected_condition:
                raise ValueError(f"{path}: Latin-order invariant failed")
            if condition == "COPY_TO_THEN_COPY_TO":
                expected = ("dpu_copy_to", "TO_DPU", "0", "0")
            elif condition == "COPY_FROM_THEN_COPY_TO":
                expected = ("dpu_copy_from", "FROM_DPU", "1", "0")
            else:
                expected = ("dpu_copy_from", "FROM_DPU", "1", "1")
            actual = (
                row["previous_op"],
                row["previous_direction"],
                row["direction_switched"],
                row["after_launch"],
            )
            if actual != expected:
                raise ValueError(
                    f"{path}: predecessor semantics for {condition} are {actual}, "
                    f"expected {expected}"
                )
            if condition == "LAUNCH_COPY_FROM_THEN_COPY_TO":
                if int(row["launch_ns"]) <= 0:
                    raise ValueError(f"{path}: launch condition has zero launch_ns")
            elif int(row["launch_ns"]) != 0:
                raise ValueError(f"{path}: non-launch condition has launch_ns")
            by_sample[sample_index].append(row)
            rows.append(row)
            source_hashes_by_size[row["transfer_bytes_per_dpu"]].add(
                row["source_content_hash"]
            )

        expected_samples = list(range(len(by_sample)))
        if sorted(by_sample) != expected_samples:
            raise ValueError(f"{path}: sample indexes are not contiguous from zero")
        for sample_index, sample_rows in by_sample.items():
            observed = sorted(row["condition"] for row in sample_rows)
            if observed != sorted(CONDITIONS):
                raise ValueError(
                    f"{path}: sample {sample_index} has conditions {observed}"
                )
    for transfer_bytes, hashes in source_hashes_by_size.items():
        if len(hashes) != 1:
            raise ValueError(
                f"transfer size {transfer_bytes}: controlled source hashes changed"
            )
    return rows


def summarize_conditions(
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
            row["transfer_bytes_per_dpu"],
            row["offset_bytes"],
            row["condition"],
        )
        grouped[key].append((int(row["measured_ns"]), row["_trace_file"]))

    output: list[dict[str, object]] = []
    for key, samples in sorted(grouped.items()):
        values = [sample[0] for sample in samples]
        trace_files = len({sample[1] for sample in samples})
        stats = distribution(values)
        if len(values) < min_samples or trace_files < min_traces:
            stability = "INSUFFICIENT"
        elif (
            float(stats["spread_pct"]) <= spread_threshold_pct
            and float(stats["cv_pct"]) <= cv_threshold_pct
        ):
            stability = "STABLE"
        else:
            stability = "UNSTABLE"
        output.append(
            {
                "configured_dpus": key[0],
                "num_tasklets": key[1],
                "target_global_dpu_id": key[2],
                "rank_ordinal": key[3],
                "dpu_id_in_rank": key[4],
                "transfer_bytes_per_dpu": key[5],
                "offset_bytes": key[6],
                "condition": key[7],
                "trace_files": trace_files,
                **stats,
                "stability": stability,
            }
        )
    return output


def effect_class(
    median_pct: float, p10_delta: float, p90_delta: float, threshold_pct: float
) -> str:
    if p10_delta > 0:
        if median_pct >= threshold_pct:
            return "CONSISTENT_SLOWER"
        return "CONSISTENT_SMALL_SLOWER"
    if p90_delta < 0:
        if median_pct <= -threshold_pct:
            return "CONSISTENT_FASTER"
        return "CONSISTENT_SMALL_FASTER"
    return "MIXED"


def summarize_pairs(
    rows: list[dict[str, str]], effect_threshold_pct: float
) -> list[dict[str, object]]:
    paired: dict[tuple[str, str, str, str], dict[str, int]] = defaultdict(dict)
    metadata: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (
            row["_trace_file"],
            row["process_repeat"],
            row["target_global_dpu_id"],
            row["sample_index"],
        )
        paired[key][row["condition"]] = int(row["measured_ns"])
        metadata[key] = row

    comparisons = (
        (
            "DIRECTION_SWITCH_EFFECT",
            "COPY_FROM_THEN_COPY_TO",
            "COPY_TO_THEN_COPY_TO",
        ),
        (
            "AFTER_LAUNCH_EFFECT",
            "LAUNCH_COPY_FROM_THEN_COPY_TO",
            "COPY_FROM_THEN_COPY_TO",
        ),
        (
            "COMBINED_ITERATIVE_HISTORY_EFFECT",
            "LAUNCH_COPY_FROM_THEN_COPY_TO",
            "COPY_TO_THEN_COPY_TO",
        ),
    )
    grouped: dict[tuple[str, ...], list[tuple[float, float]]] = defaultdict(list)
    for key, values in paired.items():
        row = metadata[key]
        for effect, lhs, rhs in comparisons:
            delta = float(values[lhs] - values[rhs])
            delta_pct = 0.0 if values[rhs] == 0 else delta / values[rhs] * 100.0
            group_key = (
                row["configured_dpus"],
                row["num_tasklets"],
                row["target_global_dpu_id"],
                row["rank_ordinal"],
                row["dpu_id_in_rank"],
                effect,
                lhs,
                rhs,
            )
            grouped[group_key].append((delta, delta_pct))

    output: list[dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        deltas = [item[0] for item in values]
        delta_pcts = [item[1] for item in values]
        median_pct = statistics.median(delta_pcts)
        p10_delta = quantile(deltas, 0.10)
        p90_delta = quantile(deltas, 0.90)
        output.append(
            {
                "configured_dpus": key[0],
                "num_tasklets": key[1],
                "target_global_dpu_id": key[2],
                "rank_ordinal": key[3],
                "dpu_id_in_rank": key[4],
                "effect": key[5],
                "lhs_condition": key[6],
                "rhs_condition": key[7],
                "paired_n": len(values),
                "median_delta_ns": statistics.median(deltas),
                "p10_delta_ns": p10_delta,
                "p90_delta_ns": p90_delta,
                "median_delta_pct": median_pct,
                "positive_pair_pct": sum(delta > 0 for delta in deltas)
                / len(deltas)
                * 100.0,
                "effect_class": effect_class(
                    median_pct, p10_delta, p90_delta, effect_threshold_pct
                ),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--paired-output", type=Path, required=True)
    parser.add_argument("--effect-threshold-pct", type=float, default=5.0)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--min-traces", type=int, default=5)
    parser.add_argument("--spread-threshold-pct", type=float, default=25.0)
    parser.add_argument("--cv-threshold-pct", type=float, default=25.0)
    args = parser.parse_args()

    rows = read_and_validate(args.traces)
    summaries = summarize_conditions(
        rows,
        min_samples=args.min_samples,
        min_traces=args.min_traces,
        spread_threshold_pct=args.spread_threshold_pct,
        cv_threshold_pct=args.cv_threshold_pct,
    )
    paired = summarize_pairs(rows, args.effect_threshold_pct)
    write_csv(args.summary_output, summaries)
    write_csv(args.paired_output, paired)

    print(f"trace_files={len(args.traces)}")
    print(f"rows={len(rows)}")
    print(f"paired_cycles={len(rows) // len(CONDITIONS)}")
    print(
        "source_control="
        f"same_pointer_within_trace,content_hash={rows[0]['source_content_hash']},"
        f"alignment={rows[0]['source_alignment_bytes']},"
        f"target_precondition={rows[0]['target_precondition']}"
    )
    print("\ncondition summaries:")
    for row in summaries:
        print(
            f"  dpu={row['target_global_dpu_id']} "
            f"condition={row['condition']} n={row['n']} "
            f"traces={row['trace_files']} "
            f"median_ns={row['median_ns']:.1f} "
            f"p10_ns={row['p10_ns']:.1f} p90_ns={row['p90_ns']:.1f} "
            f"cv_pct={row['cv_pct']:.2f} spread_pct={row['spread_pct']:.2f} "
            f"stability={row['stability']}"
        )
    print("\npaired effects:")
    for row in paired:
        print(
            f"  dpu={row['target_global_dpu_id']} effect={row['effect']} "
            f"n={row['paired_n']} median_delta_ns={row['median_delta_ns']:.1f} "
            f"p10_delta_ns={row['p10_delta_ns']:.1f} "
            f"p90_delta_ns={row['p90_delta_ns']:.1f} "
            f"median_delta_pct={row['median_delta_pct']:.2f} "
            f"class={row['effect_class']}"
        )


if __name__ == "__main__":
    main()
