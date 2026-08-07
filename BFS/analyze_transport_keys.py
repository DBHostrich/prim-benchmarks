#!/usr/bin/env python3
"""Summarize repeated BFS transfer costs by canonical transport key."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from transport_key import (
    TRANSFER_OPS,
    transport_key_with_mux_domain_min,
    transport_key_with_mux_domain_allocated_topology,
    transport_key_with_mux_domain_rank_invariant,
    transport_key_with_mux_relation_min,
    sdk_topology_relation,
    transport_key,
    transport_key_v7_full,
    transport_key_v6_full,
    transport_key_with_full_context,
    transport_key_with_phase,
    transport_key_with_phase_allocated_topology,
    transport_key_without_mux_pair_context,
    transport_key_without_physical_rank,
    transport_key_without_phase,
)


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def analyze(
    paths: list[Path],
    min_samples: int,
    min_traces: int,
    spread_threshold_pct: float,
    cv_threshold_pct: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    groups: dict[str, list[tuple[Path, dict[str, str]]]] = defaultdict(list)
    baseline_groups: dict[
        str, list[tuple[Path, dict[str, str]]]
    ] = defaultdict(list)
    phase_groups: dict[
        str, list[tuple[Path, dict[str, str]]]
    ] = defaultdict(list)
    full_context_groups: dict[
        str, list[tuple[Path, dict[str, str]]]
    ] = defaultdict(list)
    history_min_groups: dict[
        str, list[tuple[Path, dict[str, str]]]
    ] = defaultdict(list)
    physical_identity_groups: dict[
        str, list[tuple[Path, dict[str, str]]]
    ] = defaultdict(list)
    mux_relation_min_groups: dict[
        str, list[tuple[Path, dict[str, str]]]
    ] = defaultdict(list)
    mux_domain_min_groups: dict[
        str, list[tuple[Path, dict[str, str]]]
    ] = defaultdict(list)
    mux_domain_rank_invariant_groups: dict[
        str, list[tuple[Path, dict[str, str]]]
    ] = defaultdict(list)
    phase_allocated_topology_groups: dict[
        str, list[tuple[Path, dict[str, str]]]
    ] = defaultdict(list)
    mux_domain_allocated_topology_groups: dict[
        str, list[tuple[Path, dict[str, str]]]
    ] = defaultdict(list)
    transfer_rows = 0

    for path in paths:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"{path}: empty trace")
        if any(
            row["op"] in TRANSFER_OPS
            and not row.get("transport_key", "").startswith(
                ("v6;", "v7;", "v8;")
            )
            for row in rows
        ):
            for index, row in enumerate(rows):
                if row["op"] in TRANSFER_OPS:
                    previous = rows[index - 1] if index else None
                    row["previous_sdk_topology_relation"] = (
                        sdk_topology_relation(previous, row)
                    )
        for row in rows:
            if row["op"] not in TRANSFER_OPS:
                continue
            if "sdk_physical_rank_id" not in row:
                row["sdk_physical_rank_id"] = "unknown"
            if "physical_dpu_identity" not in row:
                row["physical_dpu_identity"] = "unknown"
            for field in (
                "dpu_sysfs_rank_id",
                "dpu_rank_numa_node",
                "dpu_channel_id",
                "dpu_ci_id",
                "dpu_member_id",
                "cpu_dpu_numa_relation",
            ):
                if field not in row:
                    row[field] = "unknown"
            expected_key = transport_key(row)
            v7_key = transport_key_v7_full(row)
            v6_key = transport_key_v6_full(row)
            physical_identity_key = transport_key_without_mux_pair_context(row)
            history_min_key = transport_key_without_physical_rank(row)
            full_context_key = transport_key_with_full_context(row)
            if row["transport_key"] not in {
                expected_key,
                v7_key,
                v6_key,
                physical_identity_key,
                history_min_key,
                full_context_key,
            }:
                raise ValueError(
                    f"{path}: event {row.get('event_id', '?')} has invalid "
                    "transport_key"
                )
            transfer_rows += 1
            groups[expected_key].append((path, row))
            baseline_groups[transport_key_without_phase(row)].append((path, row))
            phase_groups[transport_key_with_phase(row)].append((path, row))
            full_context_groups[full_context_key].append((path, row))
            history_min_groups[history_min_key].append((path, row))
            physical_identity_groups[physical_identity_key].append((path, row))
            mux_relation_min_groups[
                transport_key_with_mux_relation_min(row)
            ].append((path, row))
            mux_domain_min_groups[
                transport_key_with_mux_domain_min(row)
            ].append((path, row))
            mux_domain_rank_invariant_groups[
                transport_key_with_mux_domain_rank_invariant(row)
            ].append((path, row))
            phase_allocated_topology_groups[
                transport_key_with_phase_allocated_topology(row)
            ].append((path, row))
            mux_domain_allocated_topology_groups[
                transport_key_with_mux_domain_allocated_topology(row)
            ].append((path, row))

    summaries: list[dict[str, object]] = []
    for key, samples in sorted(groups.items()):
        representative = samples[0][1]
        durations = [int(row["measured_ns"]) for _, row in samples]
        median = float(statistics.median(durations))
        p10 = percentile(durations, 0.10)
        p90 = percentile(durations, 0.90)
        mean = float(statistics.mean(durations))
        stdev = float(statistics.stdev(durations)) if len(durations) > 1 else 0.0
        spread_pct = 0.0 if median == 0 else 100.0 * (p90 - p10) / median
        cv_pct = 0.0 if mean == 0 else 100.0 * stdev / mean
        samples_per_trace = Counter(str(path) for path, _ in samples)
        trace_count = len(samples_per_trace)
        failed_thresholds = []
        if len(durations) < min_samples:
            failed_thresholds.append(f"samples<{min_samples}")
        if trace_count < min_traces:
            failed_thresholds.append(f"traces<{min_traces}")
        if len(durations) < min_samples or trace_count < min_traces:
            status = "insufficient"
        else:
            if spread_pct > spread_threshold_pct:
                failed_thresholds.append(
                    f"spread>{spread_threshold_pct:g}%"
                )
            if cv_pct > cv_threshold_pct:
                failed_thresholds.append(f"cv>{cv_threshold_pct:g}%")
            status = "stable" if not failed_thresholds else "unstable"

        def joined_values(field: str, numeric: bool = False) -> str:
            values = {row[field] for _, row in samples if row.get(field, "")}
            ordered = sorted(values, key=int if numeric else None)
            return "|".join(ordered)

        def numeric_range(field: str) -> tuple[str, str]:
            values = [
                int(row[field]) for _, row in samples if row.get(field, "")
            ]
            return (
                (str(min(values)), str(max(values))) if values else ("", "")
            )

        op_index_min, op_index_max = numeric_range("op_call_index")
        dpu_op_index_min, dpu_op_index_max = numeric_range(
            "dpu_op_call_index"
        )
        previous_sdk_gap_min, previous_sdk_gap_max = numeric_range(
            "ns_since_previous_sdk_event"
        )
        summaries.append(
            {
                "transport_key": key,
                "op": representative["op"],
                "direction": representative["direction"],
                "sdk_api_kind": representative["sdk_api_kind"],
                "logical_distribution_class": representative[
                    "logical_distribution_class"
                ],
                "target_space": representative["target_space"],
                "transfer_bytes_per_dpu": representative[
                    "transfer_bytes_per_dpu"
                ],
                "active_dpus": representative["active_dpus"],
                "active_ranks": representative["active_ranks"],
                "active_dpus_per_rank": representative[
                    "active_dpus_per_rank"
                ],
                "rank_ordinal": representative["rank_ordinal"],
                "dpu_id_in_rank": representative["dpu_id_in_rank"],
                "sdk_physical_rank_id": representative[
                    "sdk_physical_rank_id"
                ],
                "dpu_sysfs_rank_id": representative["dpu_sysfs_rank_id"],
                "dpu_rank_numa_node": representative[
                    "dpu_rank_numa_node"
                ],
                "dpu_channel_id": representative["dpu_channel_id"],
                "sdk_slice_id": representative["sdk_slice_id"],
                "sdk_member_id": representative["sdk_member_id"],
                "dpu_ci_id": representative["dpu_ci_id"],
                "dpu_member_id": representative["dpu_member_id"],
                "physical_dpu_identity": representative[
                    "physical_dpu_identity"
                ],
                "cpu_dpu_numa_relation": representative[
                    "cpu_dpu_numa_relation"
                ],
                "same_source_across_group": representative[
                    "same_source_across_group"
                ],
                "phase_class_values": joined_values("phase_class"),
                "previous_sdk_ops": joined_values("previous_sdk_op"),
                "previous_sdk_directions": joined_values(
                    "previous_sdk_direction"
                ),
                "previous_sdk_transfer_bytes_values": joined_values(
                    "previous_sdk_transfer_bytes", numeric=True
                ),
                "previous_sdk_topology_relations": joined_values(
                    "previous_sdk_topology_relation"
                ),
                "ns_since_previous_sdk_event_min": previous_sdk_gap_min,
                "ns_since_previous_sdk_event_max": previous_sdk_gap_max,
                "previous_dpu_direction": representative[
                    "previous_dpu_direction"
                ],
                "previous_dpu_transfer_bytes": representative[
                    "previous_dpu_transfer_bytes"
                ],
                "previous_dpu_target_relation": representative[
                    "previous_dpu_target_relation"
                ],
                "launches_since_previous_dpu_transfer": representative[
                    "launches_since_previous_dpu_transfer"
                ],
                "target_region_reuse_class": representative[
                    "target_region_reuse_class"
                ],
                "host_buffer_page_offset": representative[
                    "host_buffer_page_offset"
                ],
                "host_buffer_reuse_class": representative[
                    "host_buffer_reuse_class"
                ],
                "configured_dpus_values": joined_values(
                    "configured_dpus", numeric=True
                ),
                "num_tasklets_values": joined_values(
                    "num_tasklets", numeric=True
                ),
                "subops": joined_values("subop"),
                "bfs_levels": joined_values("bfs_level", numeric=True),
                "logical_bytes_values": joined_values(
                    "logical_bytes", numeric=True
                ),
                "offset_features": joined_values("offset_feature"),
                "process_states": joined_values("process_state"),
                "pretrace_warmup_runs_values": joined_values(
                    "pretrace_warmup_runs", numeric=True
                ),
                "host_numa_nodes": joined_values("host_numa_node"),
                "op_call_index_min": op_index_min,
                "op_call_index_max": op_index_max,
                "dpu_op_call_index_min": dpu_op_index_min,
                "dpu_op_call_index_max": dpu_op_index_max,
                "trace_count": trace_count,
                "run_repeat_count": len(
                    {
                        (row.get("run_id", ""), row["repeat_id"])
                        for _, row in samples
                    }
                ),
                "samples_per_trace_min": min(samples_per_trace.values()),
                "samples_per_trace_max": max(samples_per_trace.values()),
                "sample_count": len(durations),
                "median_ns": round(median),
                "p10_ns": round(p10),
                "p90_ns": round(p90),
                "p90_p10_spread_pct": f"{spread_pct:.3f}",
                "mean_ns": f"{mean:.3f}",
                "stdev_ns": f"{stdev:.3f}",
                "cv_pct": f"{cv_pct:.3f}",
                "min_ns": min(durations),
                "max_ns": max(durations),
                "status": status,
                "status_reason": (
                    "within_thresholds"
                    if not failed_thresholds
                    else "|".join(failed_thresholds)
                ),
            }
        )

    status_counts = Counter(row["status"] for row in summaries)
    sample_status_counts = Counter()
    for row in summaries:
        sample_status_counts[str(row["status"])] += int(row["sample_count"])
    eligible_groups = status_counts["stable"] + status_counts["unstable"]
    eligible_samples = (
        sample_status_counts["stable"] + sample_status_counts["unstable"]
    )
    stable_transport_key_pct = (
        0.0
        if eligible_groups == 0
        else 100.0 * status_counts["stable"] / eligible_groups
    )
    stable_event_pct = (
        0.0
        if eligible_samples == 0
        else 100.0 * sample_status_counts["stable"] / eligible_samples
    )

    def group_quality(
        grouped_samples: dict[str, list[tuple[Path, dict[str, str]]]],
    ) -> dict[str, float | int]:
        group_counts = Counter()
        sample_counts = Counter()
        for samples in grouped_samples.values():
            durations = [int(row["measured_ns"]) for _, row in samples]
            trace_count = len({str(path) for path, _ in samples})
            if len(durations) < min_samples or trace_count < min_traces:
                group_counts["insufficient"] += 1
                sample_counts["insufficient"] += len(durations)
                continue
            median = float(statistics.median(durations))
            p10 = percentile(durations, 0.10)
            p90 = percentile(durations, 0.90)
            mean = float(statistics.mean(durations))
            stdev = (
                float(statistics.stdev(durations))
                if len(durations) > 1
                else 0.0
            )
            spread_pct = (
                0.0 if median == 0 else 100.0 * (p90 - p10) / median
            )
            cv_pct = 0.0 if mean == 0 else 100.0 * stdev / mean
            stable = (
                spread_pct <= spread_threshold_pct
                and cv_pct <= cv_threshold_pct
            )
            status = "stable" if stable else "unstable"
            group_counts[status] += 1
            sample_counts[status] += len(durations)
        eligible_group_count = group_counts["stable"] + group_counts["unstable"]
        eligible_sample_count = (
            sample_counts["stable"] + sample_counts["unstable"]
        )
        key_pct = (
            0.0
            if eligible_group_count == 0
            else 100.0 * group_counts["stable"] / eligible_group_count
        )
        event_pct = (
            0.0
            if eligible_sample_count == 0
            else 100.0 * sample_counts["stable"] / eligible_sample_count
        )
        return {
            "stable_groups": group_counts["stable"],
            "unstable_groups": group_counts["unstable"],
            "insufficient_groups": group_counts["insufficient"],
            "insufficient_samples": sample_counts["insufficient"],
            "stable_key_pct": key_pct,
            "stable_event_pct": event_pct,
        }

    def stable_percentages(
        grouped_samples: dict[str, list[tuple[Path, dict[str, str]]]],
    ) -> tuple[float, float]:
        quality = group_quality(grouped_samples)
        return (
            float(quality["stable_key_pct"]),
            float(quality["stable_event_pct"]),
        )

    baseline_key_pct, baseline_event_pct = stable_percentages(baseline_groups)
    phase_key_pct, phase_event_pct = stable_percentages(phase_groups)
    full_context_key_pct, full_context_event_pct = stable_percentages(
        full_context_groups
    )
    history_min_key_pct, history_min_event_pct = stable_percentages(
        history_min_groups
    )
    physical_identity_key_pct, physical_identity_event_pct = stable_percentages(
        physical_identity_groups
    )
    mux_relation_min_quality = group_quality(mux_relation_min_groups)
    mux_domain_min_quality = group_quality(mux_domain_min_groups)
    mux_domain_rank_invariant_quality = group_quality(
        mux_domain_rank_invariant_groups
    )
    phase_allocated_topology_quality = group_quality(
        phase_allocated_topology_groups
    )
    mux_domain_allocated_topology_quality = group_quality(
        mux_domain_allocated_topology_groups
    )
    mixed_phase_hardware_groups = sum(
        len({row["phase_class"] for _, row in samples}) > 1
        for samples in groups.values()
    )
    mixed_phase_mux_relation_min_groups = sum(
        len({row["phase_class"] for _, row in samples}) > 1
        for samples in mux_relation_min_groups.values()
    )
    mixed_phase_mux_domain_min_groups = sum(
        len({row["phase_class"] for _, row in samples}) > 1
        for samples in mux_domain_min_groups.values()
    )
    mixed_phase_mux_domain_rank_invariant_groups = sum(
        len({row["phase_class"] for _, row in samples}) > 1
        for samples in mux_domain_rank_invariant_groups.values()
    )
    mixed_phase_mux_domain_allocated_topology_groups = sum(
        len({row["phase_class"] for _, row in samples}) > 1
        for samples in mux_domain_allocated_topology_groups.values()
    )
    overview: dict[str, object] = {
        "transport_key_version": "v8_physical_cpu_dpu_topology",
        "trace_files": len(paths),
        "transfer_rows": transfer_rows,
        "transport_key_groups": len(summaries),
        "phase_v2_groups": len(phase_groups),
        "physical_identity_v5_groups": len(physical_identity_groups),
        "history_min_v4_groups": len(history_min_groups),
        "full_context_v3_groups": len(full_context_groups),
        "baseline_12_field_groups": len(baseline_groups),
        "hardware_groups_mixing_phase_classes": mixed_phase_hardware_groups,
        "mux_relation_min_groups": len(mux_relation_min_groups),
        "mux_relation_min_groups_mixing_phase_classes": (
            mixed_phase_mux_relation_min_groups
        ),
        "mux_relation_min_stable_groups": mux_relation_min_quality[
            "stable_groups"
        ],
        "mux_relation_min_unstable_groups": mux_relation_min_quality[
            "unstable_groups"
        ],
        "mux_relation_min_insufficient_groups": mux_relation_min_quality[
            "insufficient_groups"
        ],
        "mux_relation_min_insufficient_samples": mux_relation_min_quality[
            "insufficient_samples"
        ],
        "mux_relation_min_stable_transport_key_pct": (
            f"{mux_relation_min_quality['stable_key_pct']:.3f}"
        ),
        "mux_relation_min_stable_event_pct": (
            f"{mux_relation_min_quality['stable_event_pct']:.3f}"
        ),
        "mux_domain_min_groups": len(mux_domain_min_groups),
        "mux_domain_min_groups_mixing_phase_classes": (
            mixed_phase_mux_domain_min_groups
        ),
        "mux_domain_min_stable_groups": mux_domain_min_quality[
            "stable_groups"
        ],
        "mux_domain_min_unstable_groups": mux_domain_min_quality[
            "unstable_groups"
        ],
        "mux_domain_min_insufficient_groups": mux_domain_min_quality[
            "insufficient_groups"
        ],
        "mux_domain_min_insufficient_samples": mux_domain_min_quality[
            "insufficient_samples"
        ],
        "mux_domain_min_stable_transport_key_pct": (
            f"{mux_domain_min_quality['stable_key_pct']:.3f}"
        ),
        "mux_domain_min_stable_event_pct": (
            f"{mux_domain_min_quality['stable_event_pct']:.3f}"
        ),
        "mux_domain_rank_invariant_groups": len(
            mux_domain_rank_invariant_groups
        ),
        "mux_domain_rank_invariant_groups_mixing_phase_classes": (
            mixed_phase_mux_domain_rank_invariant_groups
        ),
        "mux_domain_rank_invariant_stable_groups": (
            mux_domain_rank_invariant_quality["stable_groups"]
        ),
        "mux_domain_rank_invariant_unstable_groups": (
            mux_domain_rank_invariant_quality["unstable_groups"]
        ),
        "mux_domain_rank_invariant_insufficient_groups": (
            mux_domain_rank_invariant_quality["insufficient_groups"]
        ),
        "mux_domain_rank_invariant_insufficient_samples": (
            mux_domain_rank_invariant_quality["insufficient_samples"]
        ),
        "mux_domain_rank_invariant_stable_transport_key_pct": (
            f"{mux_domain_rank_invariant_quality['stable_key_pct']:.3f}"
        ),
        "mux_domain_rank_invariant_stable_event_pct": (
            f"{mux_domain_rank_invariant_quality['stable_event_pct']:.3f}"
        ),
        "phase_allocated_topology_groups": len(
            phase_allocated_topology_groups
        ),
        "phase_allocated_topology_stable_groups": (
            phase_allocated_topology_quality["stable_groups"]
        ),
        "phase_allocated_topology_unstable_groups": (
            phase_allocated_topology_quality["unstable_groups"]
        ),
        "phase_allocated_topology_insufficient_groups": (
            phase_allocated_topology_quality["insufficient_groups"]
        ),
        "phase_allocated_topology_insufficient_samples": (
            phase_allocated_topology_quality["insufficient_samples"]
        ),
        "phase_allocated_topology_stable_transport_key_pct": (
            f"{phase_allocated_topology_quality['stable_key_pct']:.3f}"
        ),
        "phase_allocated_topology_stable_event_pct": (
            f"{phase_allocated_topology_quality['stable_event_pct']:.3f}"
        ),
        "mux_domain_allocated_topology_groups": len(
            mux_domain_allocated_topology_groups
        ),
        "mux_domain_allocated_topology_groups_mixing_phase_classes": (
            mixed_phase_mux_domain_allocated_topology_groups
        ),
        "mux_domain_allocated_topology_stable_groups": (
            mux_domain_allocated_topology_quality["stable_groups"]
        ),
        "mux_domain_allocated_topology_unstable_groups": (
            mux_domain_allocated_topology_quality["unstable_groups"]
        ),
        "mux_domain_allocated_topology_insufficient_groups": (
            mux_domain_allocated_topology_quality["insufficient_groups"]
        ),
        "mux_domain_allocated_topology_insufficient_samples": (
            mux_domain_allocated_topology_quality["insufficient_samples"]
        ),
        "mux_domain_allocated_topology_stable_transport_key_pct": (
            f"{mux_domain_allocated_topology_quality['stable_key_pct']:.3f}"
        ),
        "mux_domain_allocated_topology_stable_event_pct": (
            f"{mux_domain_allocated_topology_quality['stable_event_pct']:.3f}"
        ),
        "stable_groups": status_counts["stable"],
        "unstable_groups": status_counts["unstable"],
        "insufficient_groups": status_counts["insufficient"],
        "stable_samples": sample_status_counts["stable"],
        "unstable_samples": sample_status_counts["unstable"],
        "insufficient_samples": sample_status_counts["insufficient"],
        "stable_transport_key_pct": f"{stable_transport_key_pct:.3f}",
        "stable_event_pct": f"{stable_event_pct:.3f}",
        "phase_v2_stable_transport_key_pct": f"{phase_key_pct:.3f}",
        "phase_v2_stable_event_pct": f"{phase_event_pct:.3f}",
        "physical_identity_v5_stable_transport_key_pct": (
            f"{physical_identity_key_pct:.3f}"
        ),
        "physical_identity_v5_stable_event_pct": (
            f"{physical_identity_event_pct:.3f}"
        ),
        "history_min_v4_stable_transport_key_pct": (
            f"{history_min_key_pct:.3f}"
        ),
        "history_min_v4_stable_event_pct": f"{history_min_event_pct:.3f}",
        "full_context_v3_stable_transport_key_pct": (
            f"{full_context_key_pct:.3f}"
        ),
        "full_context_v3_stable_event_pct": (
            f"{full_context_event_pct:.3f}"
        ),
        "baseline_12_field_stable_transport_key_pct": f"{baseline_key_pct:.3f}",
        "baseline_12_field_stable_event_pct": f"{baseline_event_pct:.3f}",
        "min_samples": min_samples,
        "min_traces": min_traces,
        "spread_threshold_pct": spread_threshold_pct,
        "cv_threshold_pct": cv_threshold_pct,
    }
    return summaries, overview


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--min-traces", type=int, default=20)
    parser.add_argument("--spread-threshold-pct", type=float, default=25.0)
    parser.add_argument("--cv-threshold-pct", type=float, default=25.0)
    args = parser.parse_args()
    if args.min_samples < 2:
        parser.error("--min-samples must be at least 2")
    if args.min_traces < 2:
        parser.error("--min-traces must be at least 2")
    if args.spread_threshold_pct < 0:
        parser.error("--spread-threshold-pct must be non-negative")
    if args.cv_threshold_pct < 0:
        parser.error("--cv-threshold-pct must be non-negative")

    summaries, overview = analyze(
        args.traces,
        args.min_samples,
        args.min_traces,
        args.spread_threshold_pct,
        args.cv_threshold_pct,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not summaries:
        raise SystemExit("No dpu_copy_to/from rows found")
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    for name, value in overview.items():
        print(f"{name}={value}")
    print(f"summary_csv={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
