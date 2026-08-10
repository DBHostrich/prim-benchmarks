#!/usr/bin/env python3
"""Analyze paired primary/replay GEMV input-vector transfers."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


CONTEXTS = ("FIRST_USE", "REUSED")
ORDINALS = ("PRIMARY", "IDENTICAL_REPLAY")


def percentile(values: List[int], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile from zero values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower
    )


def robust_threshold(values: List[int]) -> float:
    median = float(statistics.median(values))
    mad = float(statistics.median(abs(value - median) for value in values))
    return median + 3.0 * 1.4826 * mad


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


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(durations: List[int], threshold: float) -> Dict[str, object]:
    median = float(statistics.median(durations))
    p10 = percentile(durations, 0.10)
    p90 = percentile(durations, 0.90)
    mean = statistics.mean(durations)
    cv = statistics.stdev(durations) / mean if len(durations) > 1 else 0.0
    spread = (p90 - p10) / median if median else 0.0
    mad = statistics.median(abs(value - median) for value in durations)
    slow_count = sum(value > threshold for value in durations)
    return {
        "event_count": len(durations),
        "median_ns": round(median),
        "p10_ns": round(p10),
        "p90_ns": round(p90),
        "spread_pct": f"{100 * spread:.3f}",
        "cv_pct": f"{100 * cv:.3f}",
        "mad_ns": round(mad),
        "primary_baseline_slow_threshold_ns": round(threshold),
        "slow_event_count": slow_count,
        "slow_event_pct": f"{100 * slow_count / len(durations):.3f}",
        "stable_25pct": int(spread <= 0.25 and cv <= 0.25),
    }


def analyze(result_root: Path) -> Dict[str, List[Dict[str, object]]]:
    trace_paths = sorted(
        path
        for path in result_root.glob("GEMV_128dpu_16tl_VECTOR_REPLAY/trace_*.csv")
        if not path.name.endswith("_dpus.csv")
    )
    if not trace_paths:
        raise ValueError(f"no vector-replay traces under {result_root}")

    pairs: List[Dict[str, object]] = []
    for trace_path in trace_paths:
        heartbeat_path = trace_path.with_name(
            trace_path.name.replace("trace_", "heartbeat_", 1)
        )
        if not heartbeat_path.is_file():
            raise ValueError(f"missing heartbeat for {trace_path}")
        heartbeat = read_heartbeat(heartbeat_path)
        with trace_path.open(newline="") as stream:
            trace_rows = list(csv.DictReader(stream))
        if {row["vector_replay_mode"] for row in trace_rows} != {
            "IDENTICAL_REPLAY"
        }:
            raise ValueError(f"unexpected replay mode in {trace_path}")
        by_event_id = {int(row["event_id"]): row for row in trace_rows}
        vectors: Dict[int, Dict[str, Dict[str, str]]] = defaultdict(dict)
        for row in trace_rows:
            if row["op"] == "dpu_push_xfer" and row["subop"] == "input_vector":
                ordinal = row["diagnostic_copy_ordinal"]
                if ordinal not in ORDINALS:
                    raise ValueError(f"unsupported copy ordinal in {trace_path}")
                iteration = int(row["iteration"])
                if ordinal in vectors[iteration]:
                    raise ValueError(f"duplicate {ordinal} in {trace_path}")
                vectors[iteration][ordinal] = row
        if set(vectors) != {0, 1, 2, 3}:
            raise ValueError(f"incomplete replay iterations in {trace_path}")

        for iteration in sorted(vectors):
            copies = vectors[iteration]
            if set(copies) != set(ORDINALS):
                raise ValueError(f"incomplete replay pair in {trace_path}")
            primary = copies["PRIMARY"]
            replay = copies["IDENTICAL_REPLAY"]
            primary_event_id = int(primary["event_id"])
            replay_event_id = int(replay["event_id"])
            if replay_event_id != primary_event_id + 1:
                raise ValueError(f"non-adjacent replay pair in {trace_path}")
            predecessor = by_event_id[primary_event_id - 1]
            if predecessor["subop"] != "input_matrix":
                raise ValueError(f"primary predecessor is not input_matrix in {trace_path}")
            for field in (
                "direction",
                "target_space",
                "target_symbol",
                "offset_bytes",
                "transfer_bytes_per_dpu",
                "total_logical_bytes",
                "total_transfer_bytes",
                "same_source_across_group",
                "allocated_topology_signature",
            ):
                if primary[field] != replay[field]:
                    raise ValueError(f"replay pair differs in {field}: {trace_path}")
            primary_spikes = heartbeat_overlap(
                heartbeat,
                int(primary["host_start_ns"]),
                int(primary["host_end_ns"]),
            )
            replay_spikes = heartbeat_overlap(
                heartbeat,
                int(replay["host_start_ns"]),
                int(replay["host_end_ns"]),
            )
            primary_ns = int(primary["measured_ns"])
            replay_ns = int(replay["measured_ns"])
            pairs.append(
                {
                    "run_id": primary["run_id"],
                    "repeat_id": primary["repeat_id"],
                    "iteration": primary["iteration"],
                    "primary_source_buffer_reuse_class": primary[
                        "source_buffer_reuse_class"
                    ],
                    "primary_event_id": primary_event_id,
                    "replay_event_id": replay_event_id,
                    "primary_mram_push_ordinal_since_launch": primary[
                        "mram_push_ordinal_since_launch"
                    ],
                    "replay_mram_push_ordinal_since_launch": replay[
                        "mram_push_ordinal_since_launch"
                    ],
                    "primary_source_buffer_use_count_before": primary[
                        "source_buffer_use_count_before"
                    ],
                    "replay_source_buffer_use_count_before": replay[
                        "source_buffer_use_count_before"
                    ],
                    "primary_previous_sdk_subop": primary["previous_sdk_subop"],
                    "replay_previous_sdk_subop": replay["previous_sdk_subop"],
                    "primary_transport_key": primary["transport_key"],
                    "replay_transport_key": replay["transport_key"],
                    "input_matrix_measured_ns": predecessor["measured_ns"],
                    "primary_measured_ns": primary_ns,
                    "replay_measured_ns": replay_ns,
                    "replay_to_primary_ratio": f"{replay_ns / primary_ns:.6f}",
                    "primary_wall_minus_thread_cpu_ns": primary[
                        "wall_minus_thread_cpu_ns"
                    ],
                    "replay_wall_minus_thread_cpu_ns": replay[
                        "wall_minus_thread_cpu_ns"
                    ],
                    "primary_heartbeat_spike_count": primary_spikes[0],
                    "primary_heartbeat_max_lateness_ns": primary_spikes[1],
                    "replay_heartbeat_spike_count": replay_spikes[0],
                    "replay_heartbeat_max_lateness_ns": replay_spikes[1],
                    "primary_cpu_migration": int(
                        primary["cpu_id_start"] != primary["cpu_id_end"]
                    ),
                    "replay_cpu_migration": int(
                        replay["cpu_id_start"] != replay["cpu_id_end"]
                    ),
                    "primary_involuntary_context_switch_delta": primary[
                        "involuntary_context_switch_delta"
                    ],
                    "replay_involuntary_context_switch_delta": replay[
                        "involuntary_context_switch_delta"
                    ],
                    "trace_path": str(trace_path),
                    "heartbeat_path": str(heartbeat_path),
                }
            )

    pairs_by_context: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for pair in pairs:
        context = str(pair["primary_source_buffer_reuse_class"])
        if context not in CONTEXTS:
            raise ValueError(f"unsupported primary reuse context: {context}")
        pairs_by_context[context].append(pair)
    if set(pairs_by_context) != set(CONTEXTS):
        raise ValueError("missing primary reuse context")

    thresholds = {
        context: robust_threshold(
            [int(pair["primary_measured_ns"]) for pair in context_pairs]
        )
        for context, context_pairs in pairs_by_context.items()
    }
    summary_rows: List[Dict[str, object]] = []
    comparison_rows: List[Dict[str, object]] = []
    for context in CONTEXTS:
        context_pairs = pairs_by_context[context]
        threshold = thresholds[context]
        for pair in context_pairs:
            pair["primary_baseline_slow_threshold_ns"] = round(threshold)
            pair["primary_slow"] = int(
                int(pair["primary_measured_ns"]) > threshold
            )
            pair["replay_slow_under_primary_threshold"] = int(
                int(pair["replay_measured_ns"]) > threshold
            )
        for ordinal, field in (
            ("PRIMARY", "primary_measured_ns"),
            ("IDENTICAL_REPLAY", "replay_measured_ns"),
        ):
            durations = [int(pair[field]) for pair in context_pairs]
            heartbeat_field = (
                "primary_heartbeat_spike_count"
                if ordinal == "PRIMARY"
                else "replay_heartbeat_spike_count"
            )
            summary_rows.append(
                {
                    "primary_source_buffer_reuse_class": context,
                    "diagnostic_copy_ordinal": ordinal,
                    **metrics(durations, threshold),
                    "heartbeat_event_count": sum(
                        int(pair[heartbeat_field]) > 0
                        for pair in context_pairs
                    ),
                }
            )

        primary_durations = [
            int(pair["primary_measured_ns"]) for pair in context_pairs
        ]
        replay_durations = [
            int(pair["replay_measured_ns"]) for pair in context_pairs
        ]
        primary_slow_pairs = [pair for pair in context_pairs if pair["primary_slow"]]
        primary_only = sum(
            not pair["replay_slow_under_primary_threshold"]
            for pair in primary_slow_pairs
        )
        both_slow = len(primary_slow_pairs) - primary_only
        replay_only = sum(
            not pair["primary_slow"]
            and pair["replay_slow_under_primary_threshold"]
            for pair in context_pairs
        )
        neither = len(context_pairs) - primary_only - both_slow - replay_only
        both_with_heartbeat = sum(
            pair["primary_slow"]
            and pair["replay_slow_under_primary_threshold"]
            and (
                int(pair["primary_heartbeat_spike_count"]) > 0
                or int(pair["replay_heartbeat_spike_count"]) > 0
            )
            for pair in context_pairs
        )
        if len(context_pairs) < 12:
            decision = "INSUFFICIENT_PAIRED_EVENTS"
        elif len(primary_slow_pairs) < 3:
            decision = "INSUFFICIENT_PRIMARY_SLOW_EVENTS"
        elif primary_only / len(primary_slow_pairs) >= 0.75:
            decision = "SUPPORTS_TRANSIENT_PRIMARY_TRANSFER_STATE"
        elif both_slow / len(primary_slow_pairs) >= 0.5 and (
            both_slow > 0 and both_with_heartbeat / both_slow >= 0.5
        ):
            decision = "SUPPORTS_SHARED_RUNTIME_EPISODE"
        elif both_slow / len(primary_slow_pairs) >= 0.5:
            decision = "SUPPORTS_SHARED_NONCPU_TRANSFER_EPISODE"
        else:
            decision = "PAIR_EFFECT_INCONCLUSIVE"
        comparison_rows.append(
            {
                "primary_source_buffer_reuse_class": context,
                "pair_count": len(context_pairs),
                "primary_baseline_slow_threshold_ns": round(threshold),
                "primary_median_ns": round(statistics.median(primary_durations)),
                "replay_median_ns": round(statistics.median(replay_durations)),
                "median_replay_to_primary_ratio": f"{statistics.median(replay_durations) / statistics.median(primary_durations):.6f}",
                "primary_slow_count": len(primary_slow_pairs),
                "paired_primary_slow_only": primary_only,
                "paired_both_slow": both_slow,
                "paired_replay_slow_only": replay_only,
                "paired_neither_slow": neither,
                "both_slow_with_heartbeat_count": both_with_heartbeat,
                "primary_predecessor_duration_pearson": f"{correlation([int(pair['input_matrix_measured_ns']) for pair in context_pairs], primary_durations):.6f}",
                "primary_replay_duration_pearson": f"{correlation(primary_durations, replay_durations):.6f}",
                "decision": decision,
            }
        )

    return {
        "vector_replay_events": pairs,
        "vector_replay_summary": summary_rows,
        "vector_replay_comparison": comparison_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.result_root / "vector_replay_analysis"
    try:
        outputs = analyze(args.result_root)
        for name, rows in outputs.items():
            path = output_dir / f"{name}.csv"
            write_csv(path, rows)
            print(f"{name}={path}")
    except (OSError, ValueError, KeyError, ZeroDivisionError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
