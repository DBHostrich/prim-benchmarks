#!/usr/bin/env python3
"""Compare v8 BFS transport-key candidates directly from raw traces.

The implementation reads each raw CSV once, keeps compact per-key/per-trace
duration lists, and evaluates exact leave-one-trace-out median lookup errors.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from transport_key import TRANSFER_OPS


MODEL_FIELDS = {
    "v8": (),
    "v8_prev_bytes": ("previous_sdk_transfer_bytes",),
    "v8_host_page": ("host_buffer_page_offset",),
    "v8_prev_bytes_host_page": (
        "previous_sdk_transfer_bytes",
        "host_buffer_page_offset",
    ),
}

REQUIRED_FIELDS = {
    "run_id",
    "repeat_id",
    "op",
    "subop",
    "phase_class",
    "direction",
    "logical_distribution_class",
    "transfer_bytes_per_dpu",
    "configured_dpus",
    "transport_key",
    "measured_ns",
    "previous_sdk_transfer_bytes",
    "host_buffer_page_offset",
    "physical_dpu_identity",
    "dpu_channel_id",
    "dpu_sysfs_rank_id",
    "dpu_ci_id",
    "dpu_member_id",
}


@dataclass
class GroupSamples:
    base_v8_key: str
    configured_dpus: str
    op: str
    direction: str
    logical_distribution_class: str
    transfer_bytes_per_dpu: str
    physical_dpu_identity: str
    dpu_channel_id: str
    dpu_sysfs_rank_id: str
    dpu_ci_id: str
    dpu_member_id: str
    by_trace: dict[int, list[int]] = field(default_factory=dict)
    subops: set[str] = field(default_factory=set)
    phase_classes: set[str] = field(default_factory=set)
    previous_bytes: set[str] = field(default_factory=set)
    host_pages: set[str] = field(default_factory=set)

    def add(self, trace_id: int, duration_ns: int, row: dict[str, str]) -> None:
        self.by_trace.setdefault(trace_id, []).append(duration_ns)
        self.subops.add(row["subop"])
        self.phase_classes.add(row["phase_class"])
        self.previous_bytes.add(row["previous_sdk_transfer_bytes"])
        self.host_pages.add(row["host_buffer_page_offset"])

    def durations(self) -> list[int]:
        return [
            duration
            for trace_durations in self.by_trace.values()
            for duration in trace_durations
        ]


def percentile_sorted(
    ordered: list[float] | list[int], fraction: float
) -> float:
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def percentile(values: list[float] | list[int], fraction: float) -> float:
    return percentile_sorted(sorted(values), fraction)


def joined(values: Iterable[str], numeric: bool = False) -> str:
    present = {value for value in values if value != ""}
    return "|".join(sorted(present, key=int if numeric else None))


def candidate_key(
    base_v8_key: str,
    row: dict[str, str],
    fields: tuple[str, ...],
) -> tuple[str, ...]:
    return (base_v8_key,) + tuple(row[name] for name in fields)


def printable_key(key: tuple[str, ...], fields: tuple[str, ...]) -> str:
    if not fields:
        return key[0]
    suffix = ";".join(
        f"candidate_{name}={value}"
        for name, value in zip(fields, key[1:])
    )
    return f"{key[0]};{suffix}"


def new_group(row: dict[str, str], base_v8_key: str) -> GroupSamples:
    return GroupSamples(
        base_v8_key=base_v8_key,
        configured_dpus=row["configured_dpus"],
        op=row["op"],
        direction=row["direction"],
        logical_distribution_class=row["logical_distribution_class"],
        transfer_bytes_per_dpu=row["transfer_bytes_per_dpu"],
        physical_dpu_identity=row["physical_dpu_identity"],
        dpu_channel_id=row["dpu_channel_id"],
        dpu_sysfs_rank_id=row["dpu_sysfs_rank_id"],
        dpu_ci_id=row["dpu_ci_id"],
        dpu_member_id=row["dpu_member_id"],
    )


def load_groups(
    paths: list[Path],
) -> tuple[
    dict[str, dict[tuple[str, ...], GroupSamples]],
    int,
    dict[str, int],
    dict[str, int],
]:
    groups = {model: {} for model in MODEL_FIELDS}
    transfer_rows = 0
    rows_by_configuration: Counter[str] = Counter()
    traces_by_configuration: Counter[str] = Counter()
    trace_metadata: set[tuple[str, str]] = set()

    for trace_id, path in enumerate(paths):
        file_rows = 0
        file_identity: tuple[str, str] | None = None
        file_configurations: set[str] = set()
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError(f"{path}: missing CSV header")
            missing = REQUIRED_FIELDS - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"{path}: missing required fields: {sorted(missing)}"
                )
            for row in reader:
                identity = (row["run_id"], row["repeat_id"])
                if file_identity is None:
                    file_identity = identity
                elif file_identity != identity:
                    raise ValueError(
                        f"{path}: multiple run_id/repeat_id identities"
                    )
                if row["op"] not in TRANSFER_OPS:
                    continue
                base_v8_key = row["transport_key"]
                if not base_v8_key.startswith("v8;"):
                    raise ValueError(
                        f"{path}: event {row.get('event_id', '?')} "
                        "does not contain a v8 transport key"
                    )
                try:
                    duration_ns = int(row["measured_ns"])
                except ValueError as error:
                    raise ValueError(
                        f"{path}: invalid measured_ns={row['measured_ns']}"
                    ) from error
                if duration_ns <= 0:
                    raise ValueError(
                        f"{path}: event {row.get('event_id', '?')} has "
                        f"measured_ns={duration_ns}"
                    )

                for model, fields in MODEL_FIELDS.items():
                    key = candidate_key(base_v8_key, row, fields)
                    group = groups[model].get(key)
                    if group is None:
                        group = new_group(row, base_v8_key)
                        groups[model][key] = group
                    group.add(trace_id, duration_ns, row)

                transfer_rows += 1
                file_rows += 1
                rows_by_configuration[row["configured_dpus"]] += 1
                file_configurations.add(row["configured_dpus"])

        if file_identity is None or file_rows == 0:
            raise ValueError(f"{path}: contains no transfer rows")
        if file_identity in trace_metadata:
            raise ValueError(
                f"{path}: duplicate run_id/repeat_id identity {file_identity}"
            )
        trace_metadata.add(file_identity)
        if len(file_configurations) != 1:
            raise ValueError(
                f"{path}: expected one configured_dpus value, got "
                f"{sorted(file_configurations)}"
            )
        traces_by_configuration[file_configurations.pop()] += 1
        if (trace_id + 1) % 10 == 0 or trace_id + 1 == len(paths):
            print(
                f"loaded_traces={trace_id + 1}/{len(paths)} "
                f"transfer_rows={transfer_rows}",
                file=sys.stderr,
            )

    return (
        groups,
        transfer_rows,
        dict(rows_by_configuration),
        dict(traces_by_configuration),
    )


def group_quality(
    durations: list[int],
    trace_count: int,
    min_samples: int,
    min_traces: int,
    spread_threshold_pct: float,
    cv_threshold_pct: float,
) -> dict[str, object]:
    ordered = sorted(durations)
    median = percentile_sorted(ordered, 0.50)
    p10 = percentile_sorted(ordered, 0.10)
    p90 = percentile_sorted(ordered, 0.90)
    mean = float(statistics.mean(durations))
    stdev = float(statistics.stdev(durations)) if len(durations) > 1 else 0.0
    spread_pct = 0.0 if median == 0 else 100.0 * (p90 - p10) / median
    cv_pct = 0.0 if mean == 0 else 100.0 * stdev / mean
    reasons = []
    if len(durations) < min_samples:
        reasons.append(f"samples<{min_samples}")
    if trace_count < min_traces:
        reasons.append(f"traces<{min_traces}")
    if reasons:
        status = "insufficient"
    else:
        if spread_pct > spread_threshold_pct:
            reasons.append(f"spread>{spread_threshold_pct:g}%")
        if cv_pct > cv_threshold_pct:
            reasons.append(f"cv>{cv_threshold_pct:g}%")
        status = "stable" if not reasons else "unstable"
    return {
        "sample_count": len(durations),
        "trace_count": trace_count,
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
        "status_reason": "within_thresholds" if not reasons else "|".join(reasons),
    }


def error_fields(
    absolute_ns: list[float],
    absolute_pct: list[float],
    signed_pct: list[float],
) -> dict[str, str]:
    if not absolute_ns:
        return {
            "mean_abs_error_ns": "",
            "median_abs_error_ns": "",
            "p90_abs_error_ns": "",
            "mean_abs_pct_error": "",
            "median_abs_pct_error": "",
            "p90_abs_pct_error": "",
            "p95_abs_pct_error": "",
            "mean_signed_pct_error": "",
        }
    ordered_ns = sorted(absolute_ns)
    ordered_pct = sorted(absolute_pct)
    return {
        "mean_abs_error_ns": f"{statistics.mean(absolute_ns):.3f}",
        "median_abs_error_ns": f"{percentile_sorted(ordered_ns, 0.50):.3f}",
        "p90_abs_error_ns": f"{percentile_sorted(ordered_ns, 0.90):.3f}",
        "mean_abs_pct_error": f"{statistics.mean(absolute_pct):.6f}",
        "median_abs_pct_error": f"{percentile_sorted(ordered_pct, 0.50):.6f}",
        "p90_abs_pct_error": f"{percentile_sorted(ordered_pct, 0.90):.6f}",
        "p95_abs_pct_error": f"{percentile_sorted(ordered_pct, 0.95):.6f}",
        "mean_signed_pct_error": f"{statistics.mean(signed_pct):.6f}",
    }


def compare(
    paths: list[Path],
    min_samples: int = 20,
    min_traces: int = 20,
    min_training_traces: int | None = None,
    spread_threshold_pct: float = 25.0,
    cv_threshold_pct: float = 25.0,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    if len(paths) < 2:
        raise ValueError("comparison requires at least two raw traces")
    if min_training_traces is None:
        min_training_traces = max(1, min_traces - 1)
    (
        groups_by_model,
        transfer_rows,
        rows_by_configuration,
        traces_by_configuration,
    ) = load_groups(paths)
    configurations = sorted(rows_by_configuration, key=int)
    scopes = ["ALL"] + [f"{value}dpu" for value in configurations]
    summary_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    results_by_model: dict[
        str, dict[tuple[str, ...], dict[str, object]]
    ] = {}

    for model, fields in MODEL_FIELDS.items():
        model_groups = groups_by_model[model]
        status_counts = {scope: Counter() for scope in scopes}
        status_samples = {scope: Counter() for scope in scopes}
        mixed_subops = Counter()
        mixed_phases = Counter()
        evaluation_rows = Counter()
        predicted_rows = Counter()
        abs_ns: dict[str, list[float]] = {scope: [] for scope in scopes}
        abs_pct: dict[str, list[float]] = {scope: [] for scope in scopes}
        signed_pct: dict[str, list[float]] = {scope: [] for scope in scopes}
        model_results: dict[tuple[str, ...], dict[str, object]] = {}

        for key, group in model_groups.items():
            durations = group.durations()
            quality = group_quality(
                durations,
                len(group.by_trace),
                min_samples,
                min_traces,
                spread_threshold_pct,
                cv_threshold_pct,
            )
            config_scope = f"{group.configured_dpus}dpu"
            for scope in ("ALL", config_scope):
                status = str(quality["status"])
                status_counts[scope][status] += 1
                status_samples[scope][status] += len(durations)
                mixed_subops[scope] += int(len(group.subops) > 1)
                mixed_phases[scope] += int(len(group.phase_classes) > 1)
                evaluation_rows[scope] += len(durations)

            for held_out_trace, held_out_durations in group.by_trace.items():
                training_trace_count = len(group.by_trace) - 1
                if training_trace_count < min_training_traces:
                    continue
                training = [
                    duration
                    for trace_id, trace_durations in group.by_trace.items()
                    if trace_id != held_out_trace
                    for duration in trace_durations
                ]
                if not training:
                    continue
                estimate = float(statistics.median(training))
                for actual in held_out_durations:
                    error_ns = abs(actual - estimate)
                    error_pct = 100.0 * error_ns / actual
                    bias_pct = 100.0 * (estimate - actual) / actual
                    for scope in ("ALL", config_scope):
                        predicted_rows[scope] += 1
                        abs_ns[scope].append(error_ns)
                        abs_pct[scope].append(error_pct)
                        signed_pct[scope].append(bias_pct)

            result = {
                "model": model,
                "candidate_key": printable_key(key, fields),
                "base_v8_key": group.base_v8_key,
                "configured_dpus": group.configured_dpus,
                "op": group.op,
                "direction": group.direction,
                "logical_distribution_class": group.logical_distribution_class,
                "transfer_bytes_per_dpu": group.transfer_bytes_per_dpu,
                "physical_dpu_identity": group.physical_dpu_identity,
                "dpu_channel_id": group.dpu_channel_id,
                "dpu_sysfs_rank_id": group.dpu_sysfs_rank_id,
                "dpu_ci_id": group.dpu_ci_id,
                "dpu_member_id": group.dpu_member_id,
                "subops": joined(group.subops),
                "phase_classes": joined(group.phase_classes),
                "previous_sdk_transfer_bytes_values": joined(
                    group.previous_bytes, numeric=True
                ),
                "host_buffer_page_offset_values": joined(
                    group.host_pages, numeric=True
                ),
                **quality,
            }
            group_rows.append(result)
            model_results[key] = result

        results_by_model[model] = model_results
        for scope in scopes:
            eligible_groups = (
                status_counts[scope]["stable"]
                + status_counts[scope]["unstable"]
            )
            eligible_samples = (
                status_samples[scope]["stable"]
                + status_samples[scope]["unstable"]
            )
            scope_group_count = sum(status_counts[scope].values())
            base_group_count = (
                len(groups_by_model["v8"])
                if scope == "ALL"
                else sum(
                    group.configured_dpus == scope[:-3]
                    for group in groups_by_model["v8"].values()
                )
            )
            summary_rows.append(
                {
                    "model": model,
                    "scope": scope,
                    "trace_files": len(paths),
                    "scope_trace_files": (
                        len(paths)
                        if scope == "ALL"
                        else traces_by_configuration[scope[:-3]]
                    ),
                    "transfer_rows": (
                        transfer_rows
                        if scope == "ALL"
                        else rows_by_configuration[scope[:-3]]
                    ),
                    "key_groups": scope_group_count,
                    "table_growth_vs_v8_pct": (
                        "0.000"
                        if base_group_count == 0
                        else f"{100.0 * (scope_group_count - base_group_count) / base_group_count:.3f}"
                    ),
                    "stable_groups": status_counts[scope]["stable"],
                    "unstable_groups": status_counts[scope]["unstable"],
                    "insufficient_groups": status_counts[scope]["insufficient"],
                    "stable_key_pct": (
                        "0.000"
                        if eligible_groups == 0
                        else f"{100.0 * status_counts[scope]['stable'] / eligible_groups:.3f}"
                    ),
                    "stable_event_pct": (
                        "0.000"
                        if eligible_samples == 0
                        else f"{100.0 * status_samples[scope]['stable'] / eligible_samples:.3f}"
                    ),
                    "insufficient_events": status_samples[scope]["insufficient"],
                    "mixed_subop_groups": mixed_subops[scope],
                    "mixed_phase_groups": mixed_phases[scope],
                    "loo_min_training_traces": min_training_traces,
                    "loo_evaluation_rows": evaluation_rows[scope],
                    "loo_predicted_rows": predicted_rows[scope],
                    "loo_coverage_pct": (
                        "0.000000"
                        if evaluation_rows[scope] == 0
                        else f"{100.0 * predicted_rows[scope] / evaluation_rows[scope]:.6f}"
                    ),
                    **error_fields(
                        abs_ns[scope], abs_pct[scope], signed_pct[scope]
                    ),
                }
            )

        print(
            f"evaluated_model={model} key_groups={len(model_groups)}",
            file=sys.stderr,
        )

    base_results = results_by_model["v8"]
    transition_rows: list[dict[str, object]] = []
    for model in MODEL_FIELDS:
        if model == "v8":
            continue
        children: dict[str, list[dict[str, object]]] = defaultdict(list)
        for result in results_by_model[model].values():
            children[str(result["base_v8_key"])].append(result)
        for base_result in base_results.values():
            if base_result["status"] != "unstable":
                continue
            base_key = str(base_result["base_v8_key"])
            child_rows = children[base_key]
            child_counts = Counter(str(row["status"]) for row in child_rows)
            child_samples = Counter()
            for row in child_rows:
                child_samples[str(row["status"])] += int(row["sample_count"])
            if child_counts["insufficient"]:
                outcome = "coverage_lost"
            elif child_counts["unstable"] == 0:
                outcome = "resolved"
            elif child_counts["stable"]:
                outcome = "partially_resolved"
            else:
                outcome = "unresolved"
            transition_rows.append(
                {
                    "model": model,
                    "base_v8_key": base_key,
                    "configured_dpus": base_result["configured_dpus"],
                    "physical_dpu_identity": base_result[
                        "physical_dpu_identity"
                    ],
                    "subops": base_result["subops"],
                    "base_sample_count": base_result["sample_count"],
                    "base_spread_pct": base_result[
                        "p90_p10_spread_pct"
                    ],
                    "base_cv_pct": base_result["cv_pct"],
                    "child_groups": len(child_rows),
                    "stable_children": child_counts["stable"],
                    "unstable_children": child_counts["unstable"],
                    "insufficient_children": child_counts["insufficient"],
                    "stable_child_samples": child_samples["stable"],
                    "unstable_child_samples": child_samples["unstable"],
                    "insufficient_child_samples": child_samples[
                        "insufficient"
                    ],
                    "outcome": outcome,
                }
            )

    for summary_row in summary_rows:
        model = str(summary_row["model"])
        scope = str(summary_row["scope"])
        if model == "v8":
            summary_row["base_unstable_parents"] = summary_row[
                "unstable_groups"
            ]
            summary_row["resolved_parents"] = ""
            summary_row["partially_resolved_parents"] = ""
            summary_row["unresolved_parents"] = ""
            summary_row["coverage_lost_parents"] = ""
            summary_row["resolved_parent_pct"] = ""
            continue
        relevant_transitions = [
            row
            for row in transition_rows
            if row["model"] == model
            and (
                scope == "ALL"
                or f"{row['configured_dpus']}dpu" == scope
            )
        ]
        outcomes = Counter(
            str(row["outcome"]) for row in relevant_transitions
        )
        summary_row["base_unstable_parents"] = len(relevant_transitions)
        summary_row["resolved_parents"] = outcomes["resolved"]
        summary_row["partially_resolved_parents"] = outcomes[
            "partially_resolved"
        ]
        summary_row["unresolved_parents"] = outcomes["unresolved"]
        summary_row["coverage_lost_parents"] = outcomes["coverage_lost"]
        summary_row["resolved_parent_pct"] = (
            "0.000"
            if not relevant_transitions
            else f"{100.0 * outcomes['resolved'] / len(relevant_transitions):.3f}"
        )

    summary_rows.sort(
        key=lambda row: (
            list(MODEL_FIELDS).index(str(row["model"])),
            0 if row["scope"] == "ALL" else int(str(row["scope"])[:-3]),
        )
    )
    group_rows.sort(
        key=lambda row: (
            list(MODEL_FIELDS).index(str(row["model"])),
            int(row["configured_dpus"]),
            str(row["candidate_key"]),
        )
    )
    transition_rows.sort(
        key=lambda row: (
            list(MODEL_FIELDS).index(str(row["model"])),
            int(row["configured_dpus"]),
            str(row["base_v8_key"]),
        )
    )
    return summary_rows, group_rows, transition_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.touch()
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--group-output", required=True, type=Path)
    parser.add_argument("--transition-output", required=True, type=Path)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--min-traces", type=int, default=20)
    parser.add_argument("--min-training-traces", type=int)
    parser.add_argument("--spread-threshold-pct", type=float, default=25.0)
    parser.add_argument("--cv-threshold-pct", type=float, default=25.0)
    args = parser.parse_args()

    try:
        summary, groups, transitions = compare(
            args.traces,
            min_samples=args.min_samples,
            min_traces=args.min_traces,
            min_training_traces=args.min_training_traces,
            spread_threshold_pct=args.spread_threshold_pct,
            cv_threshold_pct=args.cv_threshold_pct,
        )
        write_csv(args.summary_output, summary)
        write_csv(args.group_output, groups)
        write_csv(args.transition_output, transitions)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))

    for row in summary:
        if row["scope"] != "ALL":
            continue
        print(
            f"model={row['model']} groups={row['key_groups']} "
            f"growth_pct={row['table_growth_vs_v8_pct']} "
            f"stable_key_pct={row['stable_key_pct']} "
            f"stable_event_pct={row['stable_event_pct']} "
            f"insufficient_groups={row['insufficient_groups']} "
            f"resolved_parents={row['resolved_parents']} "
            f"loo_coverage_pct={row['loo_coverage_pct']} "
            f"loo_median_ape_pct={row['median_abs_pct_error']} "
            f"loo_p90_ape_pct={row['p90_abs_pct_error']}"
        )
    print(f"summary_csv={args.summary_output}")
    print(f"group_csv={args.group_output}")
    print(f"transition_csv={args.transition_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
