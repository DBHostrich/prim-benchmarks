#!/usr/bin/env python3
"""Select distinct physical NUMA-local CPUs for GEMV and its stall probe."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Set, Tuple


def parse_cpu_list(text: str) -> List[int]:
    result: Set[int] = set()
    for token in text.strip().split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            first_text, last_text = token.split("-", 1)
            first, last = int(first_text), int(last_text)
            if last < first:
                raise ValueError(f"descending CPU range: {token}")
            result.update(range(first, last + 1))
        else:
            result.add(int(token))
    if not result:
        raise ValueError("CPU list is empty")
    return sorted(result)


def select_cpus(
    sysfs_root: Path,
    numa_node: int,
    allowed_cpus: Optional[Set[int]] = None,
) -> Tuple[int, int, str]:
    node_path = sysfs_root / "devices/system/node" / f"node{numa_node}"
    cpu_text = (node_path / "cpulist").read_text()
    online_path = sysfs_root / "devices/system/cpu/online"
    online = set(parse_cpu_list(online_path.read_text()))
    if allowed_cpus is not None:
        online &= allowed_cpus
    representatives = {}
    for cpu_id in parse_cpu_list(cpu_text):
        if cpu_id not in online:
            continue
        topology = sysfs_root / "devices/system/cpu" / f"cpu{cpu_id}" / "topology"
        package_id = int((topology / "physical_package_id").read_text())
        core_id = int((topology / "core_id").read_text())
        representatives.setdefault((package_id, core_id), cpu_id)
    physical_cpus = sorted(representatives.values())
    if len(physical_cpus) < 2:
        raise ValueError(
            f"NUMA node {numa_node} has fewer than two online physical cores"
        )
    host_cpu = physical_cpus[0]
    probe_cpu = physical_cpus[-1]
    multi_core_cpus = [cpu for cpu in physical_cpus if cpu != probe_cpu]
    return host_cpu, probe_cpu, ",".join(str(cpu) for cpu in multi_core_cpus)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numa-node", type=int, default=0)
    parser.add_argument("--sysfs-root", type=Path, default=Path("/sys"))
    args = parser.parse_args()
    try:
        host_cpu, probe_cpu, multi_core_cpu_list = select_cpus(
            args.sysfs_root, args.numa_node, set(os.sched_getaffinity(0))
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(host_cpu, probe_cpu, multi_core_cpu_list, sep="\t")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
