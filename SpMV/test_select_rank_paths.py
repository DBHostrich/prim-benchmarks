from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from select_rank_paths import selected_rows


class SelectRankPathsTest(unittest.TestCase):
    def test_round_robin_and_exclusion(self) -> None:
        content = (
            "rank_path\tsysfs_rank_id\tsdk_rank_id\tnuma\tchannel\tci_count\t"
            "dpus_per_ci\tdpus_per_rank\tmram_per_dpu_bytes\t"
            "mram_per_rank_bytes\tstatus\n"
            "/dev/dpu_rank0\t0\t100\t0\t1\t8\t8\t64\t1\t64\tok\n"
            "/dev/dpu_rank1\t1\t101\t0\t1\t8\t8\t64\t1\t64\tok\n"
            "/dev/dpu_rank4\t4\t104\t0\t2\t8\t8\t64\t1\t64\tok\n"
            "/dev/dpu_rank5\t5\t105\t0\t2\t8\t8\t64\t1\t64\tok\n"
            "/dev/dpu_rank20\t20\t120\t1\t7\t8\t8\t64\t1\t64\tok\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topology.tsv"
            path.write_text(content)
            rows = selected_rows(path, numa_node=0, excluded_sysfs_ranks={5})
        self.assertEqual(
            [row["sysfs_rank_id"] for row in rows], ["0", "4", "1"]
        )


if __name__ == "__main__":
    unittest.main()
