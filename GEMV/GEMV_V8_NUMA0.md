# GEMV v8 NUMA0 stability experiment

This experiment measures GEMV collection transfers at 64, 128, 256, 512,
1024, and 1216 DPUs. It pins the host process and host memory to NUMA0. Rank5
is excluded from every allocation.

## Measurement semantics

Each `dpu_push_xfer` call produces one event row. Its `measured_ns` covers the
push call itself. The preceding `dpu_prepare_xfer` loop stays outside this
interval. The event row carries the complete allocated rank and channel layout.
A companion DPU CSV records each participant's logical and physical bytes.

The four transfer classes are:

| subop | direction | distribution | source reuse |
| --- | --- | --- | --- |
| `input_arguments` | `TO_DPU` | `PARTITIONED_SCATTER` | per-DPU source |
| `input_matrix` | `TO_DPU` | `PARTITIONED_SCATTER` | per-DPU source |
| `input_vector` | `TO_DPU` | `SHARED_REPLICATION` | shared source |
| `output_vector` | `FROM_DPU` | `PARTITIONED_GATHER` | per-DPU destination |

The runner uses nested channel-round-robin rank sets. With the supplied NUMA0
topology, the 1216-DPU allocation order is:

```text
0,4,8,12,16,1,6,9,13,17,2,7,10,14,18,3,11,15,19
```

Every smaller allocation is a prefix of this sequence. This keeps the scale
sets nested while spreading early ranks across channels.

## Short gate

```bash
cd ~/bdang/prim-benchmarks/GEMV

export DPU_RANK_TOPOLOGY_TSV="$HOME/upmem_topology_20260807_172654/dpu_rank_topology.tsv"

DPUS_LIST="64 128" \
PROCESS_WARMUP_RUNS=1 \
TRACE_RUNS=2 \
TRANSPORT_KEY_MIN_SAMPLES=2 \
TRANSPORT_KEY_MIN_TRACES=2 \
bash ./run_interleaved_scale_trace.sh
```

Check the printed result path, the two `validation.log` files, and
`scale_stability_summary.csv` before the formal collection.

## Formal collection

```bash
cd ~/bdang/prim-benchmarks/GEMV

export DPU_RANK_TOPOLOGY_TSV="$HOME/upmem_topology_20260807_172654/dpu_rank_topology.tsv"

bash ./run_interleaved_scale_trace.sh
```

The default run performs three process-level warmups and thirty traced fresh
processes per scale. Each process uses one in-process warmup iteration followed
by three measured iterations. Scale order rotates by round to reduce batch and
time drift.

## Results

The default result root is:

```text
/tmp/bdang/gemv_v8_numa0_<timestamp>
```

Important files include:

| path | purpose |
| --- | --- |
| `schedule.csv` | interleaved execution order and wall-clock interval |
| `GEMV_*dpu_16tl/trace_*.csv` | collection event rows and v8 keys |
| `GEMV_*dpu_16tl/trace_*_dpus.csv` | per-DPU topology and byte details |
| `GEMV_*dpu_16tl/validation.log` | semantic, topology, and byte checks |
| `GEMV_*dpu_16tl/transport_key_summary.csv` | per-key stability plus warmup and iterative medians |
| `transport_key_summary_all.csv` | all six scales in one table |
| `scale_stability_summary.csv` | one final status row per DPU scale |

The default stability gate requires at least twenty samples from at least
twenty trace files. A key is stable when `(P90-P10)/median` and CV are each at
most 25 percent. The per-scale status is stable when every eligible key passes.
