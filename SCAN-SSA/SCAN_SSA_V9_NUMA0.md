# SCAN-SSA v9 NUMA0 稳定性实验

## 实验定义

本实验把 GEMV v9 collection-transfer 标签移植到 SCAN-SSA。正式分析同时比较五种键：v8、v8 加 warmup、v8 加 `previous_sdk_op_class`、v8 加 endpoint reuse，以及完整 v9。

固定参数如下：

* DPU 数量为 64、128、256、512、1024、1152。
* 主机 CPU 和内存绑定 NUMA0，DPU 物理 rank 也固定在 NUMA0。
* `/dev/dpu_rank4` 和 `/dev/dpu_rank5` 始终排除。rank4 的 SCAN 输出校验失败，rank5 已有 MRAM/MUX 超时记录。
* SCAN 使用强缩放，总输入为 251,658,240 个 INT64，约 2 GiB。
* 每组配置先运行 3 个进程预热，再采集 30 个新进程 trace。
* 每个 trace 包含 1 次进程内预热和 3 次正式迭代。
* 稳定阈值为 `(P90-P10)/median <= 25%` 且 `CV <= 25%`，每组至少覆盖 20 个 trace。

1152 DPU 对应 NUMA0 的 18 个健康整 rank。选择器采用 channel round-robin 顺序：

```text
0,6,8,12,16,1,7,9,13,17,2,10,14,18,3,11,15,19
```

NUMA0 的 1216 DPU 配置需要 19 个整 rank。当前健康集合包含 18 个整 rank，因此正式 NUMA0 健康满载点定义为 1152 DPU。1216 DPU 可在 rank4 修复并重新通过 SCAN 正确性门禁后恢复。

每次迭代记录五类传输：

| subop | 方向 | 空间 | 数据分布 |
|---|---|---|---|
| `input_arguments_scan` | TO_DPU | WRAM | SHARED_REPLICATION |
| `input_data` | TO_DPU | MRAM | PARTITIONED_SCATTER |
| `partial_results` | FROM_DPU | WRAM | PARTITIONED_GATHER |
| `input_arguments_add` | TO_DPU | WRAM | PARTITIONED_SCATTER |
| `output_data` | FROM_DPU | MRAM | PARTITIONED_GATHER |

每个 trace 应产生 31 条 SDK 事件，其中包含 20 条 collection transfer。逐 DPU 明细保存在配套的 `trace_*_dpus.csv`。

reuse 计数按逻辑主机端点和物理 DPU 区域分别定义。五类主机端点在每轮各出现一次，所以 `source_buffer_use_count_before` 等于迭代号。两次参数传输共同访问 `DPU_INPUT_ARGUMENTS` 的相同 WRAM 区域，所以 `input_arguments_scan` 的目标计数为 `2 * iteration`，`input_arguments_add` 的目标计数为 `2 * iteration + 1`。第一轮的 add 参数因此已经属于 `target_region_reuse_class=REUSED`。

## 本机提交到 GitHub

在本机执行：

```bash
cd /home/bdang/prim-benchmarks
git status --short -- SCAN-SSA
git diff --check -- SCAN-SSA
git diff -- SCAN-SSA
git add SCAN-SSA/Makefile \
  SCAN-SSA/host/app.c \
  SCAN-SSA/host/host_trace.c \
  SCAN-SSA/host/host_trace.h \
  SCAN-SSA/transport_key.py \
  SCAN-SSA/select_rank_paths.py \
  SCAN-SSA/validate_hw_trace.py \
  SCAN-SSA/analyze_transport_keys.py \
  SCAN-SSA/analyze_context_keys.py \
  SCAN-SSA/summarize_scale_stability.py \
  SCAN-SSA/run_interleaved_scale_trace.sh \
  SCAN-SSA/test_transport_key.py \
  SCAN-SSA/test_select_rank_paths.py \
  SCAN-SSA/test_validate_hw_trace.py \
  SCAN-SSA/test_analyze_transport_keys.py \
  SCAN-SSA/test_analyze_context_keys.py \
  SCAN-SSA/test_summarize_scale_stability.py \
  SCAN-SSA/SCAN_SSA_V9_NUMA0.md
git commit -m "Fix SCAN v9 endpoint reuse accounting"
git push bdang "$(git branch --show-current)"
git rev-parse HEAD
```

记下最后一条命令打印的提交号。

## 真机从 GitHub 同步

在 `upmemcloud5` 上执行：

```bash
cd ~/bdang/prim-benchmarks
pwd
git status --short
git fetch bdang
git switch gemv-vector-replay-delay-probe
git merge --ff-only bdang/gemv-vector-replay-delay-probe
git rev-parse HEAD
```

核对真机提交号与本机提交号一致。工作区状态适合保留到结果元数据中，runner 会写入 `prim_git_status.txt`。

## 真机本地门禁

```bash
cd ~/bdang/prim-benchmarks/SCAN-SSA
pwd
python3 --version
python3 -m unittest discover -p 'test_*.py'
bash -n run_interleaved_scale_trace.sh

export DPU_RANK_TOPOLOGY_TSV="$HOME/upmem_topology_20260807_172654/dpu_rank_topology.tsv"
sha256sum "$DPU_RANK_TOPOLOGY_TSV"

python3 select_rank_paths.py "$DPU_RANK_TOPOLOGY_TSV" \
  --numa-node 0 --exclude-sysfs-ranks 4,5 \
  --rank-count 18 --field sysfs_rank_id
```

最后一条命令应打印：

```text
0,6,8,12,16,1,7,9,13,17,2,10,14,18,3,11,15,19
```

## 64、128 和 1152 DPU 小门禁

```bash
cd ~/bdang/prim-benchmarks/SCAN-SSA
GATE_STAMP=$(date +%Y%m%d_%H%M%S)
export RESULT_ROOT="/tmp/bdang/scan_ssa_v9_gate_${GATE_STAMP}"
export DPU_RANK_TOPOLOGY_TSV="$HOME/upmem_topology_20260807_172654/dpu_rank_topology.tsv"

DPUS_LIST="64 128 1152" \
PROCESS_WARMUP_RUNS=1 \
TRACE_RUNS=2 \
TRANSPORT_KEY_MIN_SAMPLES=2 \
TRANSPORT_KEY_MIN_TRACES=2 \
CREATE_ARCHIVE=1 \
./run_interleaved_scale_trace.sh
```

门禁完成后检查：

```bash
cat /tmp/bdang/latest_scan_ssa_v9_numa0_path.txt
find "$RESULT_ROOT" -name validation.log -print -exec tail -n 2 {} \;
column -t -s, "$RESULT_ROOT/scale_stability_summary.csv"
sha256sum "${RESULT_ROOT}.tar.gz"
```

## 六规模正式采集

```bash
cd ~/bdang/prim-benchmarks/SCAN-SSA
RUN_STAMP=$(date +%Y%m%d_%H%M%S)
export RESULT_ROOT="/tmp/bdang/scan_ssa_v9_numa0_${RUN_STAMP}"
export DRIVER_LOG="${RESULT_ROOT}_driver.log"
export DPU_RANK_TOPOLOGY_TSV="$HOME/upmem_topology_20260807_172654/dpu_rank_topology.tsv"

nohup env CREATE_ARCHIVE=1 ./run_interleaved_scale_trace.sh \
  > "$DRIVER_LOG" 2>&1 &
echo "$!" | tee "${RESULT_ROOT}_pid.txt"
echo "$RESULT_ROOT"
echo "$DRIVER_LOG"
```

查看进度：

```bash
tail -n 80 "$DRIVER_LOG"
find "$RESULT_ROOT" -maxdepth 2 -name 'trace_*.csv' \
  ! -name '*_dpus.csv' | wc -l
```

正式采集完成时，trace 文件数量应为 180。随后执行：

```bash
ROOT=$(cat /tmp/bdang/latest_scan_ssa_v9_numa0_path.txt)
tail -n 40 "${ROOT}_driver.log"
column -t -s, "$ROOT/scale_stability_summary.csv"
cat "$ROOT/context_analysis.log"
sha256sum "$ROOT/dpu_rank_topology.tsv" "${ROOT}.tar.gz"
```

核心结果文件如下：

* `scale_stability_summary.csv` 汇总六个规模的完整 v9 稳定性，其中 `stable_key_pct` 按键计数，`stable_event_pct` 按样本计数。
* `transport_key_summary_all.csv` 给出每个完整 v9 key 的 median、P10、P90、CV 和状态。
* `context_analysis/context_group_summary.csv` 比较五种键的稳定覆盖率和表规模。
* `context_analysis/holdout_summary.csv` 给出留一 trace 预测误差、coverage 和 signed bias。
* `dpu_rank_topology.tsv` 与对应 SHA-256 固化本次物理拓扑。

`/tmp` 属于临时存储。实验完成后及时保存 `${ROOT}.tar.gz` 及其 SHA-256。
