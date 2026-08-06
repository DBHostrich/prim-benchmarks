# BFS CPU-DPU transfer context probe

这个 probe 测量同一条 `dpu_copy_to` 在三种前序操作下的 API latency：

```text
COPY_TO_THEN_COPY_TO
COPY_FROM_THEN_COPY_TO
LAUNCH_COPY_FROM_THEN_COPY_TO
```

三种条件固定以下变量：

```text
target DPU
MRAM offset
transfer bytes
measured source pointer
measured source contents
source alignment
source cache pretouch
target contents before the predecessor chain
```

每轮使用轮换顺序执行三种条件。每次 measured copy 完成后都会读回并验证内容。
CSV 记录 source pointer、FNV-1a content hash、4KB alignment 和目标预置状态，分析器会强制检查这些控制变量。

结果根目录必须位于 `/dev/shm`。快速 gate：

```bash
cd ~/bdang/prim-benchmarks/BFS

RESULT_ROOT=/dev/shm/bfs_context_probe_gate_$(date +%Y%m%d_%H%M%S) \
NR_DPUS=256 \
NR_TASKLETS=1 \
TARGET_DPU_LIST="0" \
N_PROCESS_REPS=2 \
SAMPLES_PER_PROCESS=10 \
WARMUPS_PER_PROCESS=3 \
MIN_TRACES=2 \
MIN_SAMPLES=20 \
./run_context_probe.sh
```

完整采样建议覆盖 rank 起点、rank 终点和中间 DPU：

```bash
RESULT_ROOT=/dev/shm/bfs_context_probe_full_$(date +%Y%m%d_%H%M%S) \
NR_DPUS=256 \
NR_TASKLETS=1 \
TARGET_DPU_LIST="0 31 63 64 95 127 128 159 191 192 223 255" \
N_PROCESS_REPS=10 \
SAMPLES_PER_PROCESS=60 \
WARMUPS_PER_PROCESS=5 \
MIN_TRACES=5 \
MIN_SAMPLES=20 \
./run_context_probe.sh
```

主要结果：

```text
condition_summary.csv
paired_effects.csv
analysis.log
```

`paired_effects.csv` 中的比较含义：

```text
DIRECTION_SWITCH_EFFECT
  COPY_FROM_THEN_COPY_TO 相对 COPY_TO_THEN_COPY_TO

AFTER_LAUNCH_EFFECT
  LAUNCH_COPY_FROM_THEN_COPY_TO 相对 COPY_FROM_THEN_COPY_TO

COMBINED_ITERATIVE_HISTORY_EFFECT
  LAUNCH_COPY_FROM_THEN_COPY_TO 相对 COPY_TO_THEN_COPY_TO
```

`CONSISTENT_SLOWER` 表示中位差超过阈值，同时 P10 delta 仍大于零。分析采用配对样本，三种条件共享相同的 sample index。
