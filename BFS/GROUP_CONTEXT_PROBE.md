# BFS full-group transfer context probe

该实验把原来的单 DPU 控制实验扩展为完整 DPU sweep，测量下面三种条件：

```text
H2D_GROUP_THEN_H2D_GROUP
D2H_GROUP_THEN_H2D_GROUP
LAUNCH_D2H_GROUP_THEN_H2D_GROUP
```

每个 measured group 都按 allocation 内的固定顺序，对全部 DPU 逐个调用同步
`dpu_copy_to`。每条调用使用同一个 4KB 对齐的 host buffer，并保持内容、传输量和
每个 DPU 的目标 MRAM offset 固定。三种条件按 sample 轮换执行，降低固定执行顺序
带来的偏差。

CSV 同时记录两种时间：

```text
measured_ns        单条 dpu_copy_to 的 API latency
measured_group_ns  完整 H2D sweep 的 wall-clock latency
```

每条 measured event 的 transport key 保持 single-copy 语义，所以
`active_dpus=1`、`active_ranks=1` 和 `active_dpus_per_rank=1`。`group_size`、
`group_index` 与 `rank_boundary_before` 描述事件所在的逻辑 sweep。

结果目录固定放在 `/dev/shm`。快速 gate：

```bash
cd ~/bdang/prim-benchmarks/BFS

RESULT_ROOT=/dev/shm/bfs_group_context_probe_gate_$(date +%Y%m%d_%H%M%S) \
NR_DPUS=256 \
NR_TASKLETS=1 \
NUMA_NODE=0 \
N_PROCESS_REPS=2 \
SAMPLES_PER_PROCESS=3 \
WARMUPS_PER_PROCESS=1 \
MIN_TRACES=2 \
MIN_SAMPLES=6 \
./run_group_context_probe.sh
```

完整采样：

```bash
RESULT_ROOT=/dev/shm/bfs_group_context_probe_full_$(date +%Y%m%d_%H%M%S) \
NR_DPUS=256 \
NR_TASKLETS=1 \
NUMA_NODE=0 \
N_PROCESS_REPS=5 \
SAMPLES_PER_PROCESS=20 \
WARMUPS_PER_PROCESS=3 \
MIN_TRACES=5 \
MIN_SAMPLES=20 \
./run_group_context_probe.sh
```

分析结果：

```text
group_condition_summary.csv
group_paired_effects.csv
per_dpu_condition_summary.csv
per_dpu_paired_effects.csv
analysis.log
```

`group_paired_effects.csv` 直接回答完整的 D2H sweep 与 launch 历史能否改变下一组
H2D 的总耗时。`per_dpu_paired_effects.csv` 用于定位影响集中在哪些 group index、
rank 边界或 DPU 位置。
