from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from select_rank_paths import selected_rows


HEADER = (
    "rank_path\tsysfs_rank_id\tsdk_rank_id\tnuma\tchannel\tci_count\t"
    "dpus_per_ci\tdpus_per_rank\tmram_per_dpu_bytes\t"
    "mram_per_rank_bytes\tstatus\n"
)


class SelectRankPathsTests(unittest.TestCase):
    def test_numa0_excludes_rank4_and_rank5_and_balances_channels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topology.tsv"
            lines = [HEADER]
            channels = (
                [1, 1, 1, 1]
                + [2, 2, 2, 2]
                + [3, 3, 3, 3]
                + [4, 4, 4, 4]
                + [5, 5, 5, 5]
            )
            for rank, channel in enumerate(channels):
                lines.append(
                    f"/dev/dpu_rank{rank}\t{rank}\t{12288 + rank}\t0\t"
                    f"{channel}\t8\t8\t64\t67108864\t4294967296\tok\n"
                )
            path.write_text("".join(lines))
            rows = selected_rows(path, 0, {4, 5})
        self.assertEqual(
            [int(row["sysfs_rank_id"]) for row in rows],
            [0, 6, 8, 12, 16, 1, 7, 9, 13, 17, 2, 10, 14, 18, 3, 11, 15, 19],
        )
        self.assertTrue(
            {int(row["sysfs_rank_id"]) for row in rows}.isdisjoint({4, 5})
        )


if __name__ == "__main__":
    unittest.main()
