#!/usr/bin/env python3
"""Relate GEMV transfer outliers to host scheduling and heartbeat stalls."""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


Row = Dict[str, str]
Event = Dict[str, object]
GroupKey = Tuple[str, str, str, str]


def robust_threshold(values: List[int], multiplier: float = 3.0) -> float:
    if not values:
        raise ValueError("cannot compute a threshold from zero values")
    median = float(statistics.median(values))
    mad = float(statistics.median(abs(value - median) for value in values))
    return median + multiplier * 1.4826 * mad


def event_name(row: Row) -> str:
    return row["subop"] if row["op"] == "dpu_push_xfer" else row["op"]


def event_context(row: Row) -> str:
    if row["op"] == "dpu_push_xfer":
        return row["source_buffer_reuse_class"]
    return "FIRST" if row["warmup"] == "1" else "REUSED"


def group_key(row: Row) -> GroupKey:
    predecessor = (
        row["previous_sdk_op_class"]
        if row["op"] == "dpu_push_xfer"
        else "NOT_APPLICABLE"
    )
    return row["active_dpus"] or row["configured_dpus"], event_name(row), predecessor, event_context(row)


def is_target(row: Row) -> bool:
    if row["op"] != "dpu_push_xfer":
        return False
    if row["active_dpus"] == "128" and row["subop"] == "input_vector":
        return True
    return (
        row["active_dpus"] == "1024"
        and row["subop"] == "input_arguments"
        and row["source_buffer_reuse_class"] == "REUSED"
    )


def read_heartbeat(path: Path) -> List[Tuple[int, int, int]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = [
        (
            int(row["planned_raw_ns"]),
            int(row["actual_raw_ns"]),
            int(row["lateness_ns"]),
        )
        for row in rows
    ]
    result.sort(key=lambda item: item[1])
    return result


def heartbeat_overlap(
    spikes: List[Tuple[int, int, int]], start_ns: int, end_ns: int, window_ns: int
) -> Tuple[int, int]:
    if not spikes:
        return 0, 0
    window_start = max(0, start_ns - window_ns)
    window_end = end_ns + window_ns
    actual_times = [item[1] for item in spikes]
    index = bisect.bisect_left(actual_times, window_start)
    count = 0
    maximum = 0
    while index < len(spikes):
        planned, actual, lateness = spikes[index]
        if planned > window_end:
            break
        if actual >= window_start and planned <= window_end:
            count += 1
            maximum = max(maximum, lateness)
        index += 1
    return count, maximum


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct(numerator: int, denominator: int) -> str:
    return "" if denominator == 0 else f"{100.0 * numerator / denominator:.3f}"


def analyze(result_root: Path, window_ns: int = 200000) -> Dict[str, List[Dict[str, object]]]:
    trace_paths = sorted(
        path
        for path in result_root.glob("GEMV_*/*trace_*.csv")
        if not path.name.endswith("_dpus.csv")
    )
    if not trace_paths:
        raise ValueError(f"no event traces under {result_root}")

    events: List[Event] = []
    baseline_values: Dict[GroupKey, List[int]] = defaultdict(list)
    baseline_wait: Dict[GroupKey, List[int]] = defaultdict(list)
    heartbeat_cache: Dict[Path, List[Tuple[int, int, int]]] = {}

    for trace_path in trace_paths:
        heartbeat_path = trace_path.with_name(
            trace_path.name.replace("trace_", "heartbeat_", 1)
        )
        if not heartbeat_path.is_file():
            raise ValueError(f"missing heartbeat for {trace_path}")
        heartbeat_cache[heartbeat_path] = read_heartbeat(heartbeat_path)
        with trace_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"empty trace: {trace_path}")
        for row in rows:
            if not row["iteration"]:
                continue
            key = group_key(row)
            measured = int(row["measured_ns"])
            wait_like = int(row["wall_minus_thread_cpu_ns"])
            if row["host_binding_mode"] == "MULTI_CORE":
                baseline_values[key].append(measured)
                baseline_wait[key].append(wait_like)
            events.append(
                {
                    "trace_path": trace_path,
                    "heartbeat_path": heartbeat_path,
                    "row": row,
                    "key": key,
                    "measured_ns": measured,
                    "wait_like_ns": wait_like,
                }
            )

    thresholds = {
        key: robust_threshold(values) for key, values in baseline_values.items()
    }
    wait_thresholds = {
        key: robust_threshold(values) for key, values in baseline_wait.items()
    }
    missing = sorted({event["key"] for event in events if event["key"] not in thresholds})
    if missing:
        raise ValueError(f"groups lack MULTI_CORE baselines: {missing}")

    by_iteration: Dict[Tuple[Path, str], List[Event]] = defaultdict(list)
    for event in events:
        row = event["row"]
        assert isinstance(row, dict)
        by_iteration[(event["trace_path"], row["iteration"])].append(event)

    event_rows: List[Dict[str, object]] = []
    for event in events:
        row = event["row"]
        assert isinstance(row, dict)
        key = event["key"]
        assert isinstance(key, tuple)
        measured = int(event["measured_ns"])
        wait_like = int(event["wait_like_ns"])
        slow = measured > thresholds[key]
        wait_excess = wait_like > wait_thresholds[key]
        spike_count, max_lateness = heartbeat_overlap(
            heartbeat_cache[event["heartbeat_path"]],
            int(row["host_start_ns"]),
            int(row["host_end_ns"]),
            window_ns,
        )
        cpu_migration = row["cpu_id_start"] != row["cpu_id_end"]
        involuntary = int(row["involuntary_context_switch_delta"]) > 0
        os_runtime_signal = spike_count > 0 or cpu_migration or involuntary
        peers = by_iteration[(event["trace_path"], row["iteration"])]
        peer_slow = any(
            peer is not event
            and int(peer["measured_ns"]) > thresholds[peer["key"]]
            for peer in peers
        )
        runtime_episode_signal = os_runtime_signal or peer_slow
        event["slow"] = slow
        event_rows.append(
            {
                "target": int(is_target(row)),
                "run_id": row["run_id"],
                "repeat_id": row["repeat_id"],
                "iteration": row["iteration"],
                "active_dpus": row["active_dpus"] or row["configured_dpus"],
                "host_binding_mode": row["host_binding_mode"],
                "host_cpu_list": row["host_cpu_list"],
                "op": row["op"],
                "subop": row["subop"],
                "previous_sdk_op_class": row["previous_sdk_op_class"],
                "source_buffer_reuse_class": row["source_buffer_reuse_class"],
                "measured_ns": measured,
                "slow_threshold_ns": round(thresholds[key]),
                "slow_event": int(slow),
                "thread_cpu_ns": row["thread_cpu_ns"],
                "wall_minus_thread_cpu_ns": wait_like,
                "wait_like_threshold_ns": round(wait_thresholds[key]),
                "wait_like_excess": int(wait_excess),
                "cpu_id_start": row["cpu_id_start"],
                "cpu_id_end": row["cpu_id_end"],
                "cpu_migration": int(cpu_migration),
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
                "heartbeat_or_preempt_signal": int(os_runtime_signal),
                "same_iteration_other_slow": int(peer_slow),
                "runtime_episode_signal": int(runtime_episode_signal),
                "trace_path": str(event["trace_path"]),
                "heartbeat_path": str(event["heartbeat_path"]),
            }
        )

    summaries: List[Dict[str, object]] = []
    target_groups: Dict[Tuple[str, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in event_rows:
        if int(row["target"]) != 1:
            continue
        target_groups[
            (
                str(row["active_dpus"]),
                str(row["subop"]),
                str(row["source_buffer_reuse_class"]),
                str(row["host_binding_mode"]),
            )
        ].append(row)
    for (scale, subop, context, binding), rows in sorted(target_groups.items()):
        slow_rows = [row for row in rows if int(row["slow_event"]) == 1]
        normal_rows = [row for row in rows if int(row["slow_event"]) == 0]
        slow_signal = sum(int(row["heartbeat_or_preempt_signal"]) for row in slow_rows)
        normal_signal = sum(
            int(row["heartbeat_or_preempt_signal"]) for row in normal_rows
        )
        slow_episode_signal = sum(
            int(row["runtime_episode_signal"]) for row in slow_rows
        )
        normal_episode_signal = sum(
            int(row["runtime_episode_signal"]) for row in normal_rows
        )
        slow_signal_rate = slow_signal / len(slow_rows) if slow_rows else 0.0
        normal_signal_rate = normal_signal / len(normal_rows) if normal_rows else 0.0
        episode_slow_rate = (
            slow_episode_signal / len(slow_rows) if slow_rows else 0.0
        )
        episode_normal_rate = (
            normal_episode_signal / len(normal_rows) if normal_rows else 0.0
        )
        risk_ratio = (
            math.inf
            if episode_slow_rate > 0 and episode_normal_rate == 0
            else 0.0
            if episode_normal_rate == 0
            else episode_slow_rate / episode_normal_rate
        )
        if len(slow_rows) < 3:
            decision = "INSUFFICIENT_SLOW_EVENTS"
        elif episode_slow_rate >= 0.70 and episode_normal_rate <= 0.20 and risk_ratio >= 3:
            decision = "SUPPORTS_RUNTIME_EPISODE"
        else:
            decision = "INCONCLUSIVE_RUNTIME_EPISODE"
        summaries.append(
            {
                "active_dpus": scale,
                "subop": subop,
                "source_buffer_reuse_class": context,
                "host_binding_mode": binding,
                "event_count": len(rows),
                "slow_event_count": len(slow_rows),
                "slow_event_pct": pct(len(slow_rows), len(rows)),
                "slow_with_heartbeat_or_preempt_count": slow_signal,
                "slow_with_heartbeat_or_preempt_pct": pct(
                    slow_signal, len(slow_rows)
                ),
                "normal_with_heartbeat_or_preempt_count": normal_signal,
                "normal_with_heartbeat_or_preempt_pct": pct(
                    normal_signal, len(normal_rows)
                ),
                "slow_with_runtime_episode_count": slow_episode_signal,
                "slow_with_runtime_episode_pct": pct(
                    slow_episode_signal, len(slow_rows)
                ),
                "normal_with_runtime_episode_count": normal_episode_signal,
                "normal_with_runtime_episode_pct": pct(
                    normal_episode_signal, len(normal_rows)
                ),
                "runtime_signal_risk_ratio": (
                    "inf" if math.isinf(risk_ratio) else f"{risk_ratio:.6f}"
                ),
                "slow_with_wait_like_excess_count": sum(
                    int(row["wait_like_excess"]) for row in slow_rows
                ),
                "slow_with_same_iteration_other_slow_count": sum(
                    int(row["same_iteration_other_slow"]) for row in slow_rows
                ),
                "decision": decision,
            }
        )

    comparisons: List[Dict[str, object]] = []
    comparison_groups: Dict[
        Tuple[str, str, str], Dict[str, List[Dict[str, object]]]
    ] = defaultdict(dict)
    for (scale, subop, context, binding), rows in target_groups.items():
        comparison_groups[(scale, subop, context)][binding] = rows
    for (scale, subop, context), by_binding in sorted(comparison_groups.items()):
        multi_rows = by_binding.get("MULTI_CORE", [])
        fixed_rows = by_binding.get("FIXED_CORE", [])
        multi_slow = sum(int(row["slow_event"]) for row in multi_rows)
        fixed_slow = sum(int(row["slow_event"]) for row in fixed_rows)
        multi_rate = multi_slow / len(multi_rows) if multi_rows else 0.0
        fixed_rate = fixed_slow / len(fixed_rows) if fixed_rows else 0.0
        if len(multi_rows) == 0 or len(fixed_rows) == 0 or multi_slow < 3:
            decision = "INSUFFICIENT_BINDING_EVIDENCE"
        elif fixed_rate <= 0.5 * multi_rate:
            decision = "FIXED_CORE_REDUCES_SLOW_RATE"
        else:
            decision = "NO_CLEAR_BINDING_EFFECT"
        comparisons.append(
            {
                "active_dpus": scale,
                "subop": subop,
                "source_buffer_reuse_class": context,
                "multi_core_event_count": len(multi_rows),
                "multi_core_slow_count": multi_slow,
                "multi_core_slow_pct": pct(multi_slow, len(multi_rows)),
                "fixed_core_event_count": len(fixed_rows),
                "fixed_core_slow_count": fixed_slow,
                "fixed_core_slow_pct": pct(fixed_slow, len(fixed_rows)),
                "fixed_to_multi_slow_rate_ratio": (
                    "inf"
                    if multi_rate == 0 and fixed_rate > 0
                    else ""
                    if multi_rate == 0
                    else f"{fixed_rate / multi_rate:.6f}"
                ),
                "decision": decision,
            }
        )

    threshold_rows = [
        {
            "active_dpus": key[0],
            "event_name": key[1],
            "previous_sdk_op_class": key[2],
            "context": key[3],
            "baseline_sample_count": len(baseline_values[key]),
            "slow_threshold_ns": round(thresholds[key]),
            "wait_like_threshold_ns": round(wait_thresholds[key]),
        }
        for key in sorted(thresholds)
    ]
    return {
        "runtime_stall_events": event_rows,
        "runtime_stall_summary": summaries,
        "runtime_stall_binding_comparison": comparisons,
        "runtime_stall_thresholds": threshold_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--heartbeat-window-ns", type=int, default=200000)
    args = parser.parse_args()
    if args.heartbeat_window_ns < 0:
        parser.error("heartbeat window must be non-negative")
    output_dir = args.output_dir or args.result_root / "runtime_stall_analysis"
    try:
        outputs = analyze(args.result_root, args.heartbeat_window_ns)
        for name, rows in outputs.items():
            write_csv(output_dir / f"{name}.csv", rows)
            print(f"{name}={output_dir / f'{name}.csv'}")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
