# GEMV controlled vector-replay delay probe

## Question

The earlier identical replay experiment showed that slow `input_vector` calls
usually remain slow in the immediately following replay. This follow-up tests
whether that transfer-side episode decays after 1 ms.

Every GEMV iteration keeps the real transfer order and writes the same input
vector to the same MRAM region twice:

```text
input_arguments -> input_matrix -> input_vector PRIMARY
                -> wait 0 or 1000 us
                -> input_vector IDENTICAL_REPLAY -> dpu_launch
```

The wait uses `nanosleep`, which yields the pinned host CPU. The analyzer also
computes the observed gap from the PRIMARY end timestamp to the replay start
timestamp. This gap includes the requested sleep and replay `dpu_prepare_xfer`.

## Balanced schedules

The runner cycles through eight four-iteration schedules:

```text
0,0,0,0
0,0,1000,1000
0,1000,0,1000
0,1000,1000,0
1000,0,0,1000
1000,0,1000,0
1000,1000,0,0
1000,1000,1000,1000
```

At every iteration, each delay arm receives the same process count. Every
adjacent previous/current delay combination also receives the same count.
`TRACE_RUNS` therefore uses a multiple of eight. The gate uses 8 traces and the
formal run uses 24 traces.

## Diagnostic fields

`GEMV_VECTOR_REPLAY_DELAY_SCHEDULE_US` supplies one delay for each of the four
iterations. The event CSV includes `replay_delay_requested_us`; the replay row
carries the selected delay and every other transfer row carries zero.

The delay, copy ordinal, MRAM push ordinal, iteration, heartbeat, and observed
gap stay outside the v9 `transport_key`. They serve as experimental diagnostics.

## Interpretation

The analyzer learns one pooled PRIMARY slow threshold per reuse context, then
compares the balanced delay arms.

* A higher recovery rate after 1000 us supports a time-decaying transfer-side
  episode.
* Similar recovery rates with slow delayed replays support a state that lasts
  beyond 1 ms.
* Different PRIMARY medians flag arm imbalance.
* Fewer than three slow PRIMARY rows in either arm yields a sample-limited
  decision.

The FIRST_USE comparison has the cleanest causal interpretation because its
PRIMARY transfer precedes every scheduled delay in a fresh process. The REUSED
comparison uses the balanced transition design to control previous-delay
history.

## Hardware gate

```bash
cd ~/bdang/prim-benchmarks/GEMV
export DPU_RANK_TOPOLOGY_TSV="$HOME/upmem_topology_20260807_172654/dpu_rank_topology.tsv"

RESULT_ROOT="/tmp/bdang/gemv_vector_replay_delay_gate_$(date +%Y%m%d_%H%M%S)" \
TRACE_RUNS=8 CREATE_ARCHIVE=1 bash ./run_vector_replay_probe.sh
```

The gate produces eight validation `PASS` lines. Each trace contains 27 event
rows, 20 transfer rows, and 2,560 DPU-detail rows. Every run log must contain
`Outputs are equal`.

## Formal collection

```bash
cd ~/bdang/prim-benchmarks/GEMV
export DPU_RANK_TOPOLOGY_TSV="$HOME/upmem_topology_20260807_172654/dpu_rank_topology.tsv"

RESULT_ROOT="/tmp/bdang/gemv_vector_replay_delay_formal_$(date +%Y%m%d_%H%M%S)" \
TRACE_RUNS=24 CREATE_ARCHIVE=1 bash ./run_vector_replay_probe.sh
```

The archive contains raw traces, schedule assignments, heartbeat calibration,
the topology snapshot and checksum, validation logs, analysis CSVs, the source
commit, and the worktree status.
