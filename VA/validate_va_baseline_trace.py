#!/usr/bin/env python3
"""Validate and consolidate real-UPMEM VA baseline traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


SCHEMA_VERSION = "upmem.va_baseline.hw.v1"
PHASES = ("SETUP", "CPU_REFERENCE", "H2D", "KERNEL", "D2H", "VERIFY", "TOTAL")
HEADER = (
    "schema_version",
    "run_id",
    "repeat_id",
    "configuration",
    "actual_dpus",
    "actual_ranks",
    "tasklets",
    "block_size_log2",
    "input_elements",
    "scaling",
    "phase",
    "duration_ns",
    "bytes_per_dpu",
    "aggregate_bytes",
    "call_sequence",
    "result_ok",
    "expected_checksum",
    "actual_checksum",
    "host_numa_node",
    "sdk_version",
    "source_sha256",
    "host_binary_sha256",
    "dpu_binary_sha256",
)
EXPECTED_CALLS = {
    "SETUP": "dpu_alloc>dpu_load",
    "CPU_REFERENCE": "vector_addition_host",
    "H2D": "prepare_args>push_args>prepare_A>push_A>prepare_B>push_B",
    "KERNEL": "dpu_launch_synchronous",
    "D2H": "prepare_C>push_C",
    "VERIFY": "elementwise_compare",
    "TOTAL": "H2D>KERNEL>D2H",
}


class ValidationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--expected-reps", type=int, default=30)
    parser.add_argument("--input-elements", type=int, default=2_621_440)
    parser.add_argument("--tasklets", type=int, default=16)
    parser.add_argument("--block-size-log2", type=int, default=10)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--provenance-root", type=Path, required=True)
    return parser.parse_args()


def fail(message: str) -> None:
    raise ValidationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_provenance(root: Path) -> dict[str, object]:
    required = (
        "uname.txt",
        "lscpu.txt",
        "numa.txt",
        "numactl_show.txt",
        "prim_git_commit.txt",
        "prim_git_status.txt",
        "dpu_compiler_version.txt",
        "dpu_sdk_flags.txt",
        "source.sha256",
        "binaries.sha256",
        "config.txt",
    )
    files: dict[str, dict[str, object]] = {}
    for name in required:
        path = root / name
        if not path.is_file():
            fail(f"provenance file is absent: {path}")
        if path.stat().st_size == 0 and name != "prim_git_status.txt":
            fail(f"provenance file is empty: {path}")
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    config = (root / "config.txt").read_text()
    for marker in (
        "INPUT_ELEMENTS=2621440",
        "TASKLETS=16",
        "BLOCK_SIZE_LOG2=10",
        "SCALING=strong",
        "VA_VALIDATION_INPUT=1",
    ):
        if marker not in config:
            fail(f"provenance config lacks {marker}")
    return {
        "status": "PASS",
        "root": str(root.resolve()),
        "files": files,
    }


def aligned_per_dpu_elements(input_elements: int, actual_dpus: int) -> int:
    value = math.ceil(input_elements / actual_dpus)
    if value * 4 % 8:
        value = (value // 8) * 8 + 8
    return value


def expected_bytes(phase: str, input_elements: int, actual_dpus: int) -> tuple[int, int]:
    payload = aligned_per_dpu_elements(input_elements, actual_dpus) * 4
    if phase == "H2D":
        per_dpu = 12 + 2 * payload
    elif phase == "D2H":
        per_dpu = payload
    elif phase == "TOTAL":
        per_dpu = 12 + 3 * payload
    else:
        per_dpu = 0
    return per_dpu, per_dpu * actual_dpus


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(paths):
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != HEADER:
                fail(f"{path}: header differs from {SCHEMA_VERSION}")
            file_rows = list(reader)
        if len(file_rows) != len(PHASES):
            fail(f"{path}: expected {len(PHASES)} phase rows, found {len(file_rows)}")
        rows.extend(file_rows)
    return rows


def validate_rows(
    rows: list[dict[str, str]],
    expected_reps: int,
    input_elements: int,
    tasklets: int,
    block_size_log2: int,
) -> dict[str, object]:
    indexed: dict[tuple[str, int, str], dict[str, str]] = {}
    by_config: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        if row["schema_version"] != SCHEMA_VERSION:
            fail(f"schema_version={row['schema_version']}")
        config = row["configuration"]
        if config not in ("single", "rank"):
            fail(f"configuration={config}")
        repeat = int(row["repeat_id"])
        phase = row["phase"]
        if phase not in PHASES:
            fail(f"phase={phase}")
        key = (config, repeat, phase)
        if key in indexed:
            fail(f"duplicate row {key}")
        indexed[key] = row
        by_config[config].append(row)

        actual_dpus = int(row["actual_dpus"])
        actual_ranks = int(row["actual_ranks"])
        if actual_dpus <= 0 or actual_ranks != 1:
            fail(f"{key}: topology dpus={actual_dpus} ranks={actual_ranks}")
        if config == "single" and actual_dpus != 1:
            fail(f"{key}: single allocation has {actual_dpus} DPUs")
        if int(row["tasklets"]) != tasklets:
            fail(f"{key}: tasklets={row['tasklets']}")
        if int(row["block_size_log2"]) != block_size_log2:
            fail(f"{key}: block_size_log2={row['block_size_log2']}")
        if int(row["input_elements"]) != input_elements or row["scaling"] != "strong":
            fail(f"{key}: workload shape differs")
        if int(row["duration_ns"]) <= 0:
            fail(f"{key}: duration_ns must be positive")
        if row["result_ok"] != "1" or row["expected_checksum"] != row["actual_checksum"]:
            fail(f"{key}: functional result mismatch")
        if row["call_sequence"] != EXPECTED_CALLS[phase]:
            fail(f"{key}: call sequence differs")
        per_dpu, aggregate = expected_bytes(phase, input_elements, actual_dpus)
        if int(row["bytes_per_dpu"]) != per_dpu or int(row["aggregate_bytes"]) != aggregate:
            fail(f"{key}: byte conservation differs")
        for field in (
            "run_id",
            "host_numa_node",
            "sdk_version",
            "source_sha256",
            "host_binary_sha256",
            "dpu_binary_sha256",
        ):
            if row[field] in ("", "unknown"):
                fail(f"{key}: {field} lacks provenance")

    manifest_configs: dict[str, object] = {}
    for config in ("single", "rank"):
        config_rows = by_config.get(config, [])
        repeats = sorted({int(row["repeat_id"]) for row in config_rows})
        if repeats != list(range(1, expected_reps + 1)):
            fail(f"{config}: repeat IDs are {repeats}")
        for repeat in repeats:
            phases = {phase for cfg, rep, phase in indexed if cfg == config and rep == repeat}
            if phases != set(PHASES):
                fail(f"{config} repeat {repeat}: phase set differs")
            total = indexed[(config, repeat, "TOTAL")]
            parts = sum(
                int(indexed[(config, repeat, phase)]["duration_ns"])
                for phase in ("H2D", "KERNEL", "D2H")
            )
            if int(total["duration_ns"]) != parts:
                fail(f"{config} repeat {repeat}: TOTAL duration differs from phase sum")

        stable_fields = (
            "actual_dpus",
            "actual_ranks",
            "tasklets",
            "block_size_log2",
            "input_elements",
            "host_numa_node",
            "sdk_version",
            "source_sha256",
            "host_binary_sha256",
            "dpu_binary_sha256",
        )
        stable = {field: sorted({row[field] for row in config_rows}) for field in stable_fields}
        for field, values in stable.items():
            if len(values) != 1:
                fail(f"{config}: {field} changed across samples: {values}")
        manifest_configs[config] = {
            field: int(values[0]) if field in {
                "actual_dpus", "actual_ranks", "tasklets", "block_size_log2", "input_elements"
            } else values[0]
            for field, values in stable.items()
        }
        manifest_configs[config]["measured_processes"] = expected_reps

    return {
        "schema_version": "upmem.va_baseline.hardware_manifest.v1",
        "status": "PASS",
        "expected_phases": list(PHASES),
        "configurations": manifest_configs,
        "row_count": len(rows),
    }


def write_outputs(
    rows: list[dict[str, str]], manifest: dict[str, object], output_csv: Path, manifest_path: Path
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: (
        row["configuration"], int(row["repeat_id"]), PHASES.index(row["phase"])
    ))
    with output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(ordered)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    try:
        rows = load_rows(args.traces)
        manifest = validate_rows(
            rows, args.expected_reps, args.input_elements, args.tasklets, args.block_size_log2
        )
        manifest["provenance"] = validate_provenance(args.provenance_root)
        write_outputs(rows, manifest, args.output_csv, args.manifest)
    except (OSError, ValueError, ValidationError) as error:
        print(f"FAIL VA_HARDWARE_TRACE {error}")
        return 1
    print(
        f"PASS VA_HARDWARE_TRACE rows={len(rows)} "
        f"csv={args.output_csv} manifest={args.manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
