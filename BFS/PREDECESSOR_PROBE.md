# BFS predecessor factorial probe

This probe tests whether the strong even/odd timing split comes from the
immediately preceding single-DPU transfer.  Every measured call keeps the same
24,576 B source buffer, target frontier region, API, direction, and transport
key.  The experiment changes two predecessor factors:

| Factor | Level 1 | Level 2 |
| --- | --- | --- |
| Predecessor target | measured DPU | paired DPU in the same rank |
| Predecessor bytes | 48 B | 24,576 B |

The paired DPU uses allocation-local ordinal `ordinal ^ 1`.  Each sample
randomizes measured target order.  The trace records both that ordinal and the
SDK-reported `slice_id` and `member_id`.

The explicit predecessor writes the `dpuVisited_m` control region.  The
measured transfer writes `dpuNextFrontier_m`.  A 24,576 B readback verifies each
measured transfer after timing ends.

The analyzer reports four paired effects:

* predecessor size with the measured DPU held fixed;
* predecessor size with the paired DPU held fixed;
* paired versus same-DPU predecessor at 24,576 B;
* paired versus same-DPU predecessor at 48 B.

Parity summaries separate `ODD_TO_EVEN` and `EVEN_TO_ODD` behavior.  Position
summaries use the randomized call position to test whether order in the rank
explains the observed split.  Stability uses P10/P90 spread and CV.  A narrow
central spread with a high CV is labeled `OUTLIER_SENSITIVE`.

Run results under `/dev/shm`:

```bash
RESULT_ROOT=/dev/shm/bfs_predecessor_probe_$(date +%Y%m%d_%H%M%S) \
NR_DPUS=256 NR_TASKLETS=1 NUMA_NODE=0 \
N_PROCESS_REPS=5 SAMPLES_PER_PROCESS=20 WARMUPS_PER_PROCESS=3 \
./run_predecessor_probe.sh
```
