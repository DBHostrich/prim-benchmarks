from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from select_host_probe_cpus import parse_cpu_list, select_cpus


class SelectHostProbeCpusTests(unittest.TestCase):
    def test_parses_ranges_and_selects_one_thread_per_core(self) -> None:
        self.assertEqual(parse_cpu_list("0-2,4,6-7"), [0, 1, 2, 4, 6, 7])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            node = root / "devices/system/node/node0"
            cpu_root = root / "devices/system/cpu"
            node.mkdir(parents=True)
            cpu_root.mkdir(parents=True)
            (node / "cpulist").write_text("0-5\n")
            (cpu_root / "online").write_text("0-5\n")
            for cpu_id, core_id in enumerate((0, 0, 1, 1, 2, 2)):
                topology = cpu_root / f"cpu{cpu_id}" / "topology"
                topology.mkdir(parents=True)
                (topology / "physical_package_id").write_text("0\n")
                (topology / "core_id").write_text(f"{core_id}\n")

            host, probe, multi = select_cpus(root, 0)

        self.assertEqual(host, 0)
        self.assertEqual(probe, 4)
        self.assertEqual(multi, "0,2")


if __name__ == "__main__":
    unittest.main()
