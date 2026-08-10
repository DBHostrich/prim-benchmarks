#!/usr/bin/env python3
"""Analyze the controlled GEMV input matrix/vector transfer-order probe."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


GroupKey = Tuple[str, str]
ORDERS = ("MATRIX_THEN_VECTOR", "VECTOR_THEN_MATRIX")
CONTEXTS = ("FIRST_USE", "REUSED")


def percentile(values: List[int], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile from zero values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (
        ordered[upper] - ordered[lower]
    ) * (position - lower)


def robust_threshold(values: List[int]) -> float:
    median = float(statistics.median(values))
    mad = float(statistics.median(abs(value - median) for value in values))
    return median + 3.0 * 1.4826 * mad


def ranks(values: List[int]) -> List[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2.0 + 1.0
        for position in range(index, end):
            result[order[position]] = rank
        index = end
    return result


def correlation(left: List[int], right: List[int]) -> float:
    if len(left) < 2 or len(left) != len(right):
        return math.nan
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    left_variance = sum((value - left_mean) ** 2 for value in left)
    right_variance = sum((value - right_mean) ** 2 for value in right)
    if left_variance == 0 or right_variance == 0:
        return math.nan
    covariance = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    return covariance / math.sqrt(left_variance * right_variance)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_heartbeat(path: Path) -> List[Tuple[int, int, int]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return sorted(
        (
            int(row["planned_raw_ns"]),
            int(row["actual_raw_ns"]),
            int(row["lateness_ns"]),
        )
        for row in rows
    )


def heartbeat_overlap(
    spikes: List[Tuple[int, int, int]], start_ns: int, end_ns: int
) -> Tuple[int, int]:
    matches = [
        lateness
        for planned, actual, lateness in spikes
        if actual >= start_ns and planned <= end_ns
    ]
    return len(matches), max(matches, default=0)


def group_metrics(
    rows: List[Dict[str, object]], threshold: float
) -> Dict[str, object]:
    durations = [int(row["measured_ns"]) for row in rows]
    median = float(statistics.median(durations))
    p10 = percentile(durations, 0.10)
    p90 = percentile(durations, 0.90)
    mean = statistics.mean(durations)
    cv = statistics.stdev(durations) / mean if len(durations) > 1 and mean else 0.0
    spread = (p90 - p10) / median if median else 0.0
    mad = statistics.median(abs(value - median) for value in durations)
    slow = [row for row in rows if int(row["measured_ns"]) > threshold]
    return {
        "event_count": len(rows),
        "median_ns": round(median),
        "p10_ns": round(p10),
        "p90_ns": round(p90),
        "spread_pct": f"{100 * spread:.3f}",
        "cv_pct": f"{100 * cv:.3f}",
        "mad_ns": round(mad),
        "baseline_slow_threshold_ns": round(threshold),
        "slow_event_count": len(slow),
        "slow_event_pct": f"{100 * len(slow) / len(rows):.3f}",
        "heartbeat_event_count": sum(
            int(row["heartbeat_spike_count"]) > 0 for row in rows
        ),
        "cpu_migration_count": sum(int(row["cpu_migration"]) for row in rows),
        "involuntary_switch_event_count": sum(
            int(row["involuntary_context_switch_delta"]) > 0 for row in rows
        ),
        "minor_fault_event_count": sum(
            int(row["minor_fault_delta"]) > 0 for row in rows
        ),
        "stable_25pct": int(spread <= 0.25 and cv <= 0.25),
    }


def analyze(result_root: Path) -> Dict[str, List[Dict[str, object]]]:
    trace_paths = sorted(
        path
        for path in result_root.glob("GEMV_128dpu_16tl_*/*trace_*.csv")
        if not path.name.endswith("_dpus.csv")
    )
    if not trace_paths:
        raise ValueError(f"no order-probe traces under {result_root}")

    groups: Dict[GroupKey, List[Dict[str, object]]] = defaultdict(list)
    for trace_path in trace_paths:
        heartbeat_path = trace_path.with_name(
            trace_path.name.replace("trace_", "heartbeat_", 1)
        )
        if not heartbeat_path.is_file():
            raise ValueError(f"missing heartbeat for {trace_path}")
        heartbeat = read_heartbeat(heartbeat_path)
        with trace_path.open(newline="") as stream:
            trace_rows = list(csv.DictReader(stream))
        order = trace_rows[0]["transfer_order_variant"]
        if order not in ORDERS:
            raise ValueError(f"unsupported transfer order in {trace_path}: {order}")
        by_event_id = {int(row["event_id"]): row for row in trace_rows}
        for row in trace_rows:
            if row["op"] != "dpu_push_xfer" or row["subop"] != "input_vector":
                continue
            context = row["source_buffer_reuse_class"]
            if context not in CONTEXTS:
                raise ValueError(f"unsupported reuse context: {context}")
            event_id = int(row["event_id"])
            previous = by_event_id[event_id - 1]
            spike_count, max_lateness = heartbeat_overlap(
                heartbeat, int(row["host_start_ns"]), int(row["host_end_ns"])
            )
            groups[(order, context)].append(
                {
                    "run_id": row["run_id"],
                    "repeat_id": row["repeat_id"],
                    "iteration": row["iteration"],
                    "transfer_order_variant": order,
                    "source_buffer_reuse_class": context,
                    "previous_sdk_subop": row["previous_sdk_subop"],
                    "previous_sdk_op_class": row["previous_sdk_op_class"],
                    "previous_sdk_transfer_bytes": row[
                        "previous_sdk_transfer_bytes"
                    ],
                    "transport_key": row["transport_key"],
                    "previous_measured_ns": previous["measured_ns"],
                    "measured_ns": row["measured_ns"],
                    "thread_cpu_ns": row["thread_cpu_ns"],
                    "wall_minus_thread_cpu_ns": row["wall_minus_thread_cpu_ns"],
                    "cpu_id_start": row["cpu_id_start"],
                    "cpu_id_end": row["cpu_id_end"],
                    "cpu_migration": int(row["cpu_id_start"] != row["cpu_id_end"]),
                    "voluntary_context_switch_delta": row[
                        "voluntary_context_switch_delta"
                    ],
                    "involuntary_context_switch_delta": row[
                        "involuntary_context_switch_delta"
                    ],
                    "minor_fault_delta": row["minor_fault_delta"],
                    "major_fault_delta": row["major_fault_delta"],
                    "heartbeat_spike_count": spike_count,
                    "heartbeat_max_lateness_ns": max_lateness,
                    "trace_path": str(trace_path),
                    "heartbeat_path": str(heartbeat_path),
                }
            )

    missing = [
        key
        for key in (
            (order, context) for order in ORDERS for context in CONTEXTS
        )
        if key not in groups
    ]
    if missing:
        raise ValueError(f"missing order/context groups: {missing}")
    thresholds = {
        context: robust_threshold(
            [
                int(row["measured_ns"])
                for row in groups[("MATRIX_THEN_VECTOR", context)]
            ]
        )
        for context in CONTEXTS
    }

    event_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    metrics_by_group: Dict[GroupKey, Dict[str, object]] = {}
    for key in sorted(groups):
        order, context = key
        threshold = thresholds[context]
        for row in groups[key]:
            row["baseline_slow_threshold_ns"] = round(threshold)
            row["slow_event"] = int(int(row["measured_ns"]) > threshold)
            event_rows.append(row)
        metrics = group_metrics(groups[key], threshold)
        metrics_by_group[key] = metrics
        predecessor = {str(row["previous_sdk_subop"]) for row in groups[key]}
        previous_values = [int(row["previous_measured_ns"]) for row in groups[key]]
        durations = [int(row["measured_ns"]) for row in groups[key]]
        summary_rows.append(
            {
                "transfer_order_variant": order,
                "source_buffer_reuse_class": context,
                "previous_sdk_subop": "|".join(sorted(predecessor)),
                **metrics,
                "predecessor_duration_pearson": (
                    f"{correlation(previous_values, durations):.6f}"
                ),
                "predecessor_duration_spearman": (
                    f"{correlation(ranks(previous_values), ranks(durations)):.6f}"
                ),
            }
        )

    comparison_rows: List[Dict[str, object]] = []
    for context in CONTEXTS:
        current = metrics_by_group[("MATRIX_THEN_VECTOR", context)]
        control = metrics_by_group[("VECTOR_THEN_MATRIX", context)]
        current_rows = groups[("MATRIX_THEN_VECTOR", context)]
        control_rows = groups[("VECTOR_THEN_MATRIX", context)]
        current_rate = int(current["slow_event_count"]) / int(
            current["event_count"]
        )
        control_rate = int(control["slow_event_count"]) / int(
            control["event_count"]
        )
        spread_ratio = (
            float(control["spread_pct"]) / float(current["spread_pct"])
            if float(current["spread_pct"])
            else math.inf
        )
        cv_ratio = (
            float(control["cv_pct"]) / float(current["cv_pct"])
            if float(current["cv_pct"])
            else math.inf
        )
        current_by_round: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        control_by_round: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for row in current_rows:
            current_by_round[str(row["repeat_id"])].append(row)
        for row in control_rows:
            control_by_round[str(row["repeat_id"])].append(row)
        paired_rounds = sorted(set(current_by_round) & set(control_by_round), key=int)
        current_only = control_only = both = neither = 0
        for repeat_id in paired_rounds:
            current_slow = any(
                int(row["measured_ns"]) > thresholds[context]
                for row in current_by_round[repeat_id]
            )
            control_slow = any(
                int(row["measured_ns"]) > thresholds[context]
                for row in control_by_round[repeat_id]
            )
            if current_slow and control_slow:
                both += 1
            elif current_slow:
                current_only += 1
            elif control_slow:
                control_only += 1
            else:
                neither += 1
        strong_dispersion_reduction = (
            (spread_ratio <= 0.65 and cv_ratio <= 0.80)
            or cv_ratio <= 0.65
        )
        strong_tail_reduction = (
            int(current["slow_event_count"]) >= 3
            and control_rate <= 0.5 * current_rate
        )
        if len(paired_rounds) < 12:
            decision = "INSUFFICIENT_PAIRED_ROUNDS"
        elif strong_dispersion_reduction and strong_tail_reduction:
            decision = "SUPPORTS_PREDECESSOR_TRANSFER_STATE"
        elif spread_ratio >= 0.80 and cv_ratio >= 0.80:
            decision = "NO_CLEAR_ORDER_EFFECT"
        else:
            decision = "ORDER_EFFECT_INCONCLUSIVE"
        comparison_rows.append(
            {
                "source_buffer_reuse_class": context,
                "baseline_slow_threshold_ns": round(thresholds[context]),
                "current_event_count": current["event_count"],
                "current_median_ns": current["median_ns"],
                "current_spread_pct": current["spread_pct"],
                "current_cv_pct": current["cv_pct"],
                "current_slow_event_count": current["slow_event_count"],
                "current_slow_event_pct": current["slow_event_pct"],
                "vector_first_event_count": control["event_count"],
                "vector_first_median_ns": control["median_ns"],
                "vector_first_spread_pct": control["spread_pct"],
                "vector_first_cv_pct": control["cv_pct"],
                "vector_first_slow_event_count": control["slow_event_count"],
                "vector_first_slow_event_pct": control["slow_event_pct"],
                "vector_first_to_current_spread_ratio": f"{spread_ratio:.6f}",
                "vector_first_to_current_cv_ratio": f"{cv_ratio:.6f}",
                "paired_round_count": len(paired_rounds),
                "paired_current_slow_only": current_only,
                "paired_vector_first_slow_only": control_only,
                "paired_both_slow": both,
                "paired_neither_slow": neither,
                "decision": decision,
            }
        )
    return {
        "transfer_order_events": event_rows,
        "transfer_order_summary": summary_rows,
        "transfer_order_comparison": comparison_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.result_root / "transfer_order_analysis"
    try:
        outputs = analyze(args.result_root)
        for name, rows in outputs.items():
            path = output_dir / f"{name}.csv"
            write_csv(path, rows)
            print(f"{name}={path}")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
