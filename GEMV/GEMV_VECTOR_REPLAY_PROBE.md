# GEMV paired input-vector replay probe

## Question

This diagnostic keeps GEMV's real `input_matrix -> input_vector` order and then
immediately writes the identical input vector to the identical MRAM region a
second time before `dpu_launch`:

```text
input_arguments -> input_matrix -> input_vector PRIMARY
                -> input_vector IDENTICAL_REPLAY -> dpu_launch
```

The replay uses the same host buffer, DPU set, target symbol, offset, and byte
count. It overwrites the vector with identical contents, so the ordinary GEMV
output equality check remains the semantic gate.

## Diagnostic labels

Set `GEMV_VECTOR_REPLAY_MODE=IDENTICAL_REPLAY` to enable the extra copy. It is
accepted only with `GEMV_TRANSFER_ORDER=MATRIX_THEN_VECTOR`.

The event CSV adds these diagnostic columns:

* `vector_replay_mode`: `NONE` or `IDENTICAL_REPLAY`;
* `diagnostic_copy_ordinal`: `PRIMARY`, `IDENTICAL_REPLAY`, or `NONE`;
* `mram_push_ordinal_since_launch`: matrix=1, primary vector=2, replay=3.

The source-buffer and target-region access counts include the replay. None of
these diagnostic fields is included in the v9 `transport_key`.

## Interpretation

For every process and iteration, the analyzer pairs PRIMARY with its immediately
following replay and applies a robust threshold learned only from PRIMARY rows.

* PRIMARY slow and replay normal supports a transient first-copy transfer state.
* Both slow with a calibrated heartbeat spike supports a shared runtime episode.
* Both slow without heartbeat supports a shared SDK/driver/rank-side episode that
  the separate CPU heartbeat cannot see.
* Fewer than three slow PRIMARY events remains sample-limited.

`vector_replay_analysis/` contains the event pairs, per-copy stability summary,
and comparison/decision table. These outcomes are diagnostic evidence and do not
automatically promote a new lookup-key field.

## Collection sizes

A gate uses three traced fresh processes. The formal default uses 24. Every
process contributes one FIRST_USE primary/replay pair and three REUSED pairs.

## Hardware gate

```bash
cd ~/bdang/prim-benchmarks/GEMV
export DPU_RANK_TOPOLOGY_TSV="$HOME/upmem_topology_20260807_172654/dpu_rank_topology.tsv"

RESULT_ROOT="/tmp/bdang/gemv_vector_replay_gate_$(date +%Y%m%d_%H%M%S)" \
TRACE_RUNS=3 CREATE_ARCHIVE=1 bash ./run_vector_replay_probe.sh
```

Require three validation `PASS` lines, 27 event rows and 2,560 DPU-detail rows
per trace, and `Outputs are equal` in every run log. A three-trace analysis is
normally sample-limited and is only a hardware/schema gate.

## Formal collection

```bash
cd ~/bdang/prim-benchmarks/GEMV
export DPU_RANK_TOPOLOGY_TSV="$HOME/upmem_topology_20260807_172654/dpu_rank_topology.tsv"

RESULT_ROOT="/tmp/bdang/gemv_vector_replay_formal_$(date +%Y%m%d_%H%M%S)" \
TRACE_RUNS=24 CREATE_ARCHIVE=1 bash ./run_vector_replay_probe.sh
```

Archive the raw traces, heartbeat calibration, topology TSV/checksum, validation
log, pair-level analysis, source commit, and dirty-worktree record together.
