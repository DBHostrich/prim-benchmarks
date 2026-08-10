#!/usr/bin/env python3
"""Analyze the balanced 0/1000-us GEMV vector replay delay probe."""

from __future__ import annotations

import argparse
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from analyze_vector_replay_probe import (
    CONTEXTS,
    analyze as analyze_replay,
    percentile,
    robust_threshold,
    write_csv,
)


def median_ratio(rows: List[Dict[str, object]]) -> float:
    return float(
        statistics.median(
            int(row["replay_measured_ns"]) / int(row["primary_measured_ns"])
            for row in rows
        )
    )


def arm_metrics(
    rows: List[Dict[str, object]], threshold: float
) -> Dict[str, object]:
    primary = [int(row["primary_measured_ns"]) for row in rows]
    replay = [int(row["replay_measured_ns"]) for row in rows]
    gaps = [int(row["replay_gap_ns"]) for row in rows]
    primary_slow = [row for row in rows if int(row["primary_measured_ns"]) > threshold]
    recovered = sum(
        int(row["replay_measured_ns"]) <= threshold for row in primary_slow
    )
    return {
        "pair_count": len(rows),
        "primary_median_ns": round(statistics.median(primary)),
        "replay_median_ns": round(statistics.median(replay)),
        "median_pair_replay_to_primary_ratio": f"{median_ratio(rows):.6f}",
        "median_replay_minus_primary_ns": round(
            statistics.median(
                replay_value - primary_value
                for replay_value, primary_value in zip(replay, primary)
            )
        ),
        "actual_gap_median_ns": round(statistics.median(gaps)),
        "actual_gap_p10_ns": round(percentile(gaps, 0.10)),
        "actual_gap_p90_ns": round(percentile(gaps, 0.90)),
        "primary_slow_count": len(primary_slow),
        "primary_slow_recovered_count": recovered,
        "primary_slow_recovery_pct": (
            f"{100 * recovered / len(primary_slow):.3f}"
            if primary_slow
            else "nan"
        ),
    }


def analyze(
    result_root: Path, expected_delays_us: tuple[int, int] = (0, 1000)
) -> Dict[str, List[Dict[str, object]]]:
    base_outputs = analyze_replay(result_root)
    pairs = base_outputs["vector_replay_events"]
    expected_delays = set(expected_delays_us)
    if len(expected_delays) != 2 or min(expected_delays) < 0:
        raise ValueError("delay probe requires two distinct nonnegative delays")
    immediate_delay, delayed_delay = sorted(expected_delays)

    observed_delays = {int(row["replay_delay_requested_us"]) for row in pairs}
    if observed_delays != expected_delays:
        raise ValueError(
            f"replay delay set differs: got {sorted(observed_delays)}, "
            f"expected {sorted(expected_delays)}"
        )
    for row in pairs:
        requested_ns = int(row["replay_delay_requested_us"]) * 1000
        if int(row["replay_gap_ns"]) + 10_000 < requested_ns:
            raise ValueError("actual replay gap is shorter than the requested delay")

    by_iteration: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for row in pairs:
        by_iteration[int(row["iteration"])].append(row)
    if set(by_iteration) != {0, 1, 2, 3}:
        raise ValueError("delay probe requires iterations 0 through 3")
    for iteration, rows in by_iteration.items():
        counts = Counter(int(row["replay_delay_requested_us"]) for row in rows)
        if set(counts) != expected_delays or len(set(counts.values())) != 1:
            raise ValueError(f"iteration {iteration} delay arms are unbalanced")

    schedules: Dict[str, Dict[int, int]] = defaultdict(dict)
    for row in pairs:
        repeat_id = str(row["repeat_id"])
        schedules[repeat_id][int(row["iteration"])] = int(
            row["replay_delay_requested_us"]
        )
    transition_counts: Counter[tuple[int, int, int]] = Counter()
    for schedule in schedules.values():
        if set(schedule) != {0, 1, 2, 3}:
            raise ValueError("trace carries an incomplete replay delay schedule")
        for iteration in (1, 2, 3):
            transition_counts[(iteration, schedule[iteration - 1], schedule[iteration])] += 1
    for iteration in (1, 2, 3):
        counts = [
            transition_counts[(iteration, previous, current)]
            for previous in sorted(expected_delays)
            for current in sorted(expected_delays)
        ]
        if min(counts) == 0 or len(set(counts)) != 1:
            raise ValueError(f"iteration {iteration} delay transitions are unbalanced")
    for row in pairs:
        repeat_id = str(row["repeat_id"])
        iteration = int(row["iteration"])
        schedule = schedules[repeat_id]
        row["previous_replay_delay_requested_us"] = (
            "NONE" if iteration == 0 else schedule[iteration - 1]
        )
        row["replay_delay_schedule_us"] = "|".join(
            str(schedule[index]) for index in range(4)
        )

    pairs_by_context: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in pairs:
        pairs_by_context[str(row["primary_source_buffer_reuse_class"])].append(row)
    thresholds = {
        context: robust_threshold(
            [int(row["primary_measured_ns"]) for row in pairs_by_context[context]]
        )
        for context in CONTEXTS
    }

    summary_rows: List[Dict[str, object]] = []
    comparison_rows: List[Dict[str, object]] = []
    iteration_rows: List[Dict[str, object]] = []
    for context in CONTEXTS:
        context_rows = pairs_by_context[context]
        threshold = thresholds[context]
        arms = {
            delay: [
                row
                for row in context_rows
                if int(row["replay_delay_requested_us"]) == delay
            ]
            for delay in sorted(expected_delays)
        }
        for delay, rows in arms.items():
            summary_rows.append(
                {
                    "primary_source_buffer_reuse_class": context,
                    "replay_delay_requested_us": delay,
                    "pooled_primary_slow_threshold_ns": round(threshold),
                    **arm_metrics(rows, threshold),
                }
            )

        immediate_rows = arms[immediate_delay]
        delayed_rows = arms[delayed_delay]
        immediate_primary_median = float(
            statistics.median(int(row["primary_measured_ns"]) for row in immediate_rows)
        )
        delayed_primary_median = float(
            statistics.median(int(row["primary_measured_ns"]) for row in delayed_rows)
        )
        pooled_primary_median = float(
            statistics.median(int(row["primary_measured_ns"]) for row in context_rows)
        )
        imbalance = abs(delayed_primary_median - immediate_primary_median) / pooled_primary_median
        slow_rows = {
            delay: [
                row
                for row in rows
                if int(row["primary_measured_ns"]) > threshold
            ]
            for delay, rows in arms.items()
        }
        recovery = {
            delay: (
                sum(int(row["replay_measured_ns"]) <= threshold for row in rows)
                / len(rows)
                if rows
                else math.nan
            )
            for delay, rows in slow_rows.items()
        }
        if imbalance > 0.15:
            decision = "PRIMARY_ARMS_IMBALANCED"
        elif min(len(slow_rows[immediate_delay]), len(slow_rows[delayed_delay])) < 3:
            decision = "INSUFFICIENT_PRIMARY_SLOW_EVENTS"
        elif (
            recovery[delayed_delay] - recovery[immediate_delay] >= 0.50
            and recovery[delayed_delay] >= 0.75
        ):
            decision = "SUPPORTS_TIME_DECAYING_TRANSFER_EPISODE"
        elif (
            abs(recovery[delayed_delay] - recovery[immediate_delay]) < 0.25
            and recovery[delayed_delay] <= 0.50
        ):
            decision = "SUPPORTS_PERSISTENT_TRANSFER_STATE"
        else:
            decision = "DELAY_EFFECT_INCONCLUSIVE"
        comparison_rows.append(
            {
                "primary_source_buffer_reuse_class": context,
                "immediate_delay_us": immediate_delay,
                "delayed_delay_us": delayed_delay,
                "pooled_primary_slow_threshold_ns": round(threshold),
                "primary_arm_median_imbalance_pct": f"{100 * imbalance:.3f}",
                "immediate_primary_slow_count": len(slow_rows[immediate_delay]),
                "delayed_primary_slow_count": len(slow_rows[delayed_delay]),
                "immediate_slow_recovery_pct": (
                    f"{100 * recovery[immediate_delay]:.3f}"
                    if not math.isnan(recovery[immediate_delay])
                    else "nan"
                ),
                "delayed_slow_recovery_pct": (
                    f"{100 * recovery[delayed_delay]:.3f}"
                    if not math.isnan(recovery[delayed_delay])
                    else "nan"
                ),
                "immediate_median_pair_ratio": f"{median_ratio(immediate_rows):.6f}",
                "delayed_median_pair_ratio": f"{median_ratio(delayed_rows):.6f}",
                "decision": decision,
            }
        )

    for iteration in sorted(by_iteration):
        for delay in sorted(expected_delays):
            rows = [
                row
                for row in by_iteration[iteration]
                if int(row["replay_delay_requested_us"]) == delay
            ]
            iteration_rows.append(
                {
                    "iteration": iteration,
                    "replay_delay_requested_us": delay,
                    "pair_count": len(rows),
                    "primary_median_ns": round(
                        statistics.median(int(row["primary_measured_ns"]) for row in rows)
                    ),
                    "replay_median_ns": round(
                        statistics.median(int(row["replay_measured_ns"]) for row in rows)
                    ),
                    "median_pair_replay_to_primary_ratio": f"{median_ratio(rows):.6f}",
                    "actual_gap_median_ns": round(
                        statistics.median(int(row["replay_gap_ns"]) for row in rows)
                    ),
                }
            )

    return {
        "vector_replay_delay_events": pairs,
        "vector_replay_delay_summary": summary_rows,
        "vector_replay_delay_iteration_summary": iteration_rows,
        "vector_replay_delay_comparison": comparison_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-delay-us", type=int, nargs=2, default=(0, 1000))
    args = parser.parse_args()
    output_dir = args.output_dir or args.result_root / "vector_replay_delay_analysis"
    try:
        outputs = analyze(args.result_root, tuple(args.expected_delay_us))
        for name, rows in outputs.items():
            path = output_dir / f"{name}.csv"
            write_csv(path, rows)
            print(f"{name}={path}")
    except (OSError, ValueError, KeyError, ZeroDivisionError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
