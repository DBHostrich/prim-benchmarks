#!/usr/bin/env python3
"""Evaluate BFS transport keys with held-out lookup prediction."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from transport_key import (
    TRANSFER_OPS,
    sdk_topology_relation,
    transport_key,
    transport_key_with_mux_domain_min,
    transport_key_with_mux_domain_rank_invariant,
    transport_key_with_mux_relation_min,
    transport_key_with_phase,
    transport_key_without_phase,
)


MODEL_KEYS = {
    "base12": (transport_key_without_phase,),
    "phase_v2": (transport_key_with_phase,),
    "mux_relation_min": (transport_key_with_mux_relation_min,),
    "mux_domain_min": (transport_key_with_mux_domain_min,),
    "mux_domain_rank_invariant": (
        transport_key_with_mux_domain_rank_invariant,
    ),
    "mux_domain_hierarchical": (
        transport_key_with_mux_domain_min,
        transport_key_with_mux_domain_rank_invariant,
    ),
    "v6_full": (transport_key,),
}
SCOPES = ("ALL_TRANSFER_EVENTS", "BASE12_PHASE_MIXED")
ERROR_FIELDS = (
    "mean_abs_error_ns",
    "median_abs_error_ns",
    "p90_abs_error_ns",
    "mean_abs_pct_error",
    "median_abs_pct_error",
    "p90_abs_pct_error",
    "p95_abs_pct_error",
    "mean_signed_pct_error",
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_traces(paths: list[Path]) -> dict[Path, list[dict[str, str]]]:
    traces: dict[Path, list[dict[str, str]]] = {}
    for path in paths:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"{path}: empty trace")
        if any(
            row["op"] in TRANSFER_OPS
            and not row.get("transport_key", "").startswith("v6;")
            for row in rows
        ):
            for index, row in enumerate(rows):
                if row["op"] in TRANSFER_OPS:
                    previous = rows[index - 1] if index else None
                    row["previous_sdk_topology_relation"] = (
                        sdk_topology_relation(previous, row)
                    )
        transfers = []
        for row in rows:
            if row["op"] not in TRANSFER_OPS:
                continue
            row.setdefault("sdk_physical_rank_id", "unknown")
            row.setdefault("physical_dpu_identity", "unknown")
            transfers.append(row)
        traces[path] = transfers
    return traces


def error_summary(
    actual: list[int], predicted: list[float]
) -> dict[str, str | int]:
    absolute_ns = [abs(value - estimate) for value, estimate in zip(actual, predicted)]
    absolute_pct = [
        100.0 * abs(value - estimate) / value
        for value, estimate in zip(actual, predicted)
    ]
    signed_pct = [
        100.0 * (estimate - value) / value
        for value, estimate in zip(actual, predicted)
    ]
    return {
        "mean_abs_error_ns": f"{statistics.mean(absolute_ns):.3f}",
        "median_abs_error_ns": f"{statistics.median(absolute_ns):.3f}",
        "p90_abs_error_ns": f"{percentile(absolute_ns, 0.90):.3f}",
        "mean_abs_pct_error": f"{statistics.mean(absolute_pct):.6f}",
        "median_abs_pct_error": f"{statistics.median(absolute_pct):.6f}",
        "p90_abs_pct_error": f"{percentile(absolute_pct, 0.90):.6f}",
        "p95_abs_pct_error": f"{percentile(absolute_pct, 0.95):.6f}",
        "mean_signed_pct_error": f"{statistics.mean(signed_pct):.6f}",
    }


def evaluate(
    paths: list[Path],
    holdout_unit: str = "trace",
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    traces = load_traces(paths)
    holdout_groups: dict[str, set[Path]] = defaultdict(set)
    for path, rows in traces.items():
        if holdout_unit == "trace":
            group = str(path)
        elif holdout_unit == "configuration":
            configurations = {
                (row["configured_dpus"], row["num_tasklets"])
                for row in rows
            }
            if len(configurations) != 1:
                raise ValueError(
                    f"{path}: expected one configured_dpus/num_tasklets pair"
                )
            configured_dpus, num_tasklets = configurations.pop()
            group = f"{configured_dpus}dpu_{num_tasklets}tl"
        else:
            raise ValueError(f"unsupported holdout unit: {holdout_unit}")
        holdout_groups[group].add(path)
    if len(holdout_groups) < 2:
        raise ValueError(
            f"{holdout_unit} evaluation requires at least 2 holdout groups"
        )
    phase_values: dict[str, set[str]] = defaultdict(set)
    for rows in traces.values():
        for row in rows:
            phase_values[transport_key_without_phase(row)].add(
                row["phase_class"]
            )
    mixed_base_keys = {
        key for key, phases in phase_values.items() if len(phases) > 1
    }

    pooled: dict[
        tuple[str, str], tuple[list[int], list[float], int, int]
    ] = {
        (model, scope): ([], [], 0, 0)
        for model in MODEL_KEYS
        for scope in SCOPES
    }
    per_holdout: list[dict[str, object]] = []

    for model, key_functions in MODEL_KEYS.items():
        for held_out_group, held_out_paths in sorted(holdout_groups.items()):
            lookups = []
            for key_function in key_functions:
                training: dict[str, list[int]] = defaultdict(list)
                for path, rows in traces.items():
                    if path in held_out_paths:
                        continue
                    for row in rows:
                        training[key_function(row)].append(
                            int(row["measured_ns"])
                        )
                lookups.append(
                    {
                        key: float(statistics.median(durations))
                        for key, durations in training.items()
                    }
                )

            for scope in SCOPES:
                test_rows = [
                    row
                    for path in sorted(held_out_paths)
                    for row in traces[path]
                    if scope == "ALL_TRANSFER_EVENTS"
                    or transport_key_without_phase(row) in mixed_base_keys
                ]
                actual = []
                predicted = []
                fallback_predictions = 0
                for row in test_rows:
                    estimate = None
                    matched_level = 0
                    for level, (key_function, lookup) in enumerate(
                        zip(key_functions, lookups)
                    ):
                        estimate = lookup.get(key_function(row))
                        if estimate is not None:
                            matched_level = level
                            break
                    if estimate is None:
                        continue
                    actual.append(int(row["measured_ns"]))
                    predicted.append(estimate)
                    fallback_predictions += int(matched_level > 0)

                (
                    pooled_actual,
                    pooled_predicted,
                    pooled_total,
                    pooled_fallback,
                ) = pooled[(model, scope)]
                pooled_actual.extend(actual)
                pooled_predicted.extend(predicted)
                pooled[(model, scope)] = (
                    pooled_actual,
                    pooled_predicted,
                    pooled_total + len(test_rows),
                    pooled_fallback + fallback_predictions,
                )
                row_summary: dict[str, object] = {
                    "model": model,
                    "scope": scope,
                    "holdout_unit": holdout_unit,
                    "held_out_group": held_out_group,
                    "held_out_trace_count": len(held_out_paths),
                    "training_primary_key_count": len(lookups[0]),
                    "training_fallback_key_count": (
                        len(lookups[1]) if len(lookups) > 1 else 0
                    ),
                    "evaluation_rows": len(test_rows),
                    "predicted_rows": len(predicted),
                    "fallback_predicted_rows": fallback_predictions,
                    "fallback_predicted_pct": (
                        "0.000000"
                        if not predicted
                        else (
                            f"{100.0 * fallback_predictions / len(predicted):.6f}"
                        )
                    ),
                    "coverage_pct": (
                        "0.000000"
                        if not test_rows
                        else f"{100.0 * len(predicted) / len(test_rows):.6f}"
                    ),
                    **{field: "" for field in ERROR_FIELDS},
                }
                if predicted:
                    row_summary.update(error_summary(actual, predicted))
                per_holdout.append(row_summary)

    summary: list[dict[str, object]] = []
    for model in MODEL_KEYS:
        for scope in SCOPES:
            actual, predicted, total, fallback_predictions = pooled[
                (model, scope)
            ]
            row: dict[str, object] = {
                "model": model,
                "scope": scope,
                "holdout_unit": holdout_unit,
                "holdout_groups": len(holdout_groups),
                "trace_files": len(paths),
                "evaluation_rows": total,
                "predicted_rows": len(predicted),
                "fallback_predicted_rows": fallback_predictions,
                "fallback_predicted_pct": (
                    "0.000000"
                    if not predicted
                    else (
                        f"{100.0 * fallback_predictions / len(predicted):.6f}"
                    )
                ),
                "coverage_pct": (
                    "0.000000"
                    if total == 0
                    else f"{100.0 * len(predicted) / total:.6f}"
                ),
                **{field: "" for field in ERROR_FIELDS},
            }
            if predicted:
                row.update(error_summary(actual, predicted))
            summary.append(row)
    return summary, per_holdout


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument(
        "--holdout-unit",
        choices=("trace", "configuration"),
        default="trace",
    )
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument(
        "--per-holdout-output",
        "--per-trace-output",
        dest="per_holdout_output",
        required=True,
        type=Path,
    )
    args = parser.parse_args()

    if len(args.traces) < 2:
        parser.error("held-out evaluation requires at least 2 traces")
    try:
        summary, per_holdout = evaluate(args.traces, args.holdout_unit)
    except ValueError as error:
        parser.error(str(error))
    write_csv(args.summary_output, summary)
    write_csv(args.per_holdout_output, per_holdout)
    print(f"holdout_unit={args.holdout_unit}")
    print(f"holdout_groups={summary[0]['holdout_groups']}")
    print(f"trace_files={len(args.traces)}")
    for row in summary:
        print(
            f"model={row['model']} scope={row['scope']} "
            f"rows={row['evaluation_rows']} coverage_pct={row['coverage_pct']} "
            f"fallback_pct={row['fallback_predicted_pct']} "
            f"median_ape_pct={row.get('median_abs_pct_error', '')} "
            f"p90_ape_pct={row.get('p90_abs_pct_error', '')} "
            f"mean_ape_pct={row.get('mean_abs_pct_error', '')} "
            f"signed_bias_pct={row.get('mean_signed_pct_error', '')}"
        )
    print(f"summary_csv={args.summary_output}")
    print(f"per_holdout_csv={args.per_holdout_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
