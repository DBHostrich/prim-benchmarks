# BFS API-order probe

该实验固定 measured frontier copy 的 transport key，只改变前后 SDK 调用顺序。

三种条件如下：

```text
CONTIGUOUS_FRONTIER_GROUP
  frontier[0] -> frontier[1] -> ...

VISITED_FRONTIER_PARAMS_PER_DPU
  visited[0] -> measured frontier[0] -> params[0]
  visited[1] -> measured frontier[1] -> params[1]

D2H_MERGE_FRONTIER_PARAMS_PER_DPU
  完整 D2H frontier readback + CPU OR
  measured frontier[0] -> params[0]
  measured frontier[1] -> params[1]
```

所有 measured frontier 事件固定以下字段：

```text
op=dpu_copy_to
direction=TO_DPU
sdk_api_kind=SINGLE_COPY
logical_distribution_class=SHARED_REPLICATION
target_space=MRAM
transfer_bytes_per_dpu=24576
active_dpus=1
active_ranks=1
active_dpus_per_rank=1
same_source_across_group=1
```

同一进程内的 source pointer、内容哈希和 4KB alignment 保持一致。每个 DPU 的
measured target 固定为 `dpuNextFrontier_m`，每轮开始前写入零值。params 控制传输
使用 8 B 对齐后的 48 B 物理大小。

时间字段分为两个口径：

```text
measured_ns       单条 measured frontier copy 的 API latency
frontier_sum_ns   256 条 measured frontier latency 之和
sequence_span_ns  包含 visited/params 控制调用的完整 sequence span
```

`frontier_sum_ns` 用于判断调用顺序对 24,576 B frontier 搬运成本的影响。
`sequence_span_ns` 用于解释真实 BFS host 阶段总时间。

快速 gate：

```bash
cd ~/bdang/prim-benchmarks/BFS

RESULT_ROOT=/dev/shm/bfs_api_order_probe_gate_$(date +%Y%m%d_%H%M%S) \
NR_DPUS=256 \
NR_TASKLETS=1 \
NUMA_NODE=0 \
N_PROCESS_REPS=2 \
SAMPLES_PER_PROCESS=3 \
WARMUPS_PER_PROCESS=1 \
MIN_TRACES=2 \
MIN_SAMPLES=6 \
./run_api_order_probe.sh
```

完整采样：

```bash
RESULT_ROOT=/dev/shm/bfs_api_order_probe_full_$(date +%Y%m%d_%H%M%S) \
NR_DPUS=256 \
NR_TASKLETS=1 \
NUMA_NODE=0 \
N_PROCESS_REPS=5 \
SAMPLES_PER_PROCESS=20 \
WARMUPS_PER_PROCESS=3 \
MIN_TRACES=5 \
MIN_SAMPLES=20 \
./run_api_order_probe.sh
```

关键结果文件：

```text
group_condition_summary.csv
group_paired_effects.csv
per_dpu_condition_summary.csv
per_dpu_paired_effects.csv
analysis.log
```

`ITERATIVE_VS_INIT_ORDER_EFFECT` 直接比较迭代式调用顺序与初始化式调用顺序。
