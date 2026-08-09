# GEMV v9 context-label experiment

GEMV v9 keeps every v8 physical topology field and adds three canonical
context fields:

| field | values used by GEMV | meaning |
| --- | --- | --- |
| `previous_sdk_op_class` | `LOAD`, `LAUNCH_SYNC`, `PUSH_XFER_TO_DPU_WRAM`, `PUSH_XFER_TO_DPU_MRAM`, `PUSH_XFER_FROM_DPU_MRAM` | physical class of the immediately preceding SDK operation |
| `source_buffer_reuse_class` | `FIRST_USE`, `REUSED` | whether the same logical source endpoint has appeared in an earlier GEMV iteration |
| `target_region_reuse_class` | `FIRST_ACCESS`, `REUSED` | whether the same logical target endpoint has appeared in an earlier GEMV iteration |

The trace also records `previous_sdk_subop`, `previous_sdk_target_space`,
`source_buffer_use_count_before`, and `target_region_access_count_before` for
validation and diagnosis. These raw counters are not part of the key.

For the current GEMV loop, both reuse classes change after iteration zero. They
are therefore correlated with warmup. v9 names the physical endpoint state,
but this collection tests predictive value rather than proving whether host
buffer registration, DPU-region state, or another SDK steady-state mechanism
causes the latency change.

## Candidate comparison

`context_analysis` evaluates these lookup tables with leave-one-trace-out
prediction:

- reconstructed `base_v8`;
- v8 plus `warmup`;
- v8 plus expanded `previous_sdk_op_class`;
- v8 plus the two endpoint-reuse classes;
- complete `base_v9`.

This comparison reports table growth, coverage, median/P90/P95 absolute error,
signed bias, and per-group stability. The anomaly outputs target the five
previously unstable groups: `input_arguments` at 256/512/1024 DPUs and
`input_vector` at 128/256 DPUs.

## Short hardware gate

```bash
cd ~/bdang/prim-benchmarks/GEMV
export DPU_RANK_TOPOLOGY_TSV="$HOME/upmem_topology_20260807_172654/dpu_rank_topology.tsv"

RESULT_ROOT="/tmp/bdang/gemv_v9_gate_$(date +%Y%m%d_%H%M%S)" \
DPUS_LIST="128 256" \
PROCESS_WARMUP_RUNS=1 TRACE_RUNS=2 \
TRANSPORT_KEY_MIN_SAMPLES=2 TRANSPORT_KEY_MIN_TRACES=2 \
CREATE_ARCHIVE=1 bash ./run_interleaved_scale_trace.sh
```

Require both configurations to print `PASS`, every run log to contain
`Outputs are equal`, and both validation logs to contain two `PASS` lines.

## Targeted formal collection

```bash
cd ~/bdang/prim-benchmarks/GEMV
export DPU_RANK_TOPOLOGY_TSV="$HOME/upmem_topology_20260807_172654/dpu_rank_topology.tsv"

RESULT_ROOT="/tmp/bdang/gemv_v9_context_$(date +%Y%m%d_%H%M%S)" \
DPUS_LIST="128 256 512 1024" \
PROCESS_WARMUP_RUNS=3 TRACE_RUNS=30 \
TRANSPORT_KEY_MIN_SAMPLES=20 TRANSPORT_KEY_MIN_TRACES=20 \
CREATE_ARCHIVE=1 bash ./run_interleaved_scale_trace.sh
```

The four-scale run is sufficient for the five target groups. Keep the v8
archive as the baseline; do not merge v8 and v9 rows into one canonical-key
summary because their schemas intentionally differ.
