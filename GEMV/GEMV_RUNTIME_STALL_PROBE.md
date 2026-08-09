# GEMV runtime stall probe

## Purpose

This experiment tests whether the remaining v9 outliers coincide with a transient
host or SDK runtime episode. It preserves the v9 transport key. Every SDK event
gains diagnostic counters, while a separate pinned CPU records heartbeat delays.

The first-stage probe uses two representative cases:

* `input_vector`, 128 DPUs, covering `FIRST_USE` and `REUSED`.
* `input_arguments`, 1024 DPUs, restricted to `REUSED` by the analyzer.

Each scale runs in `MULTI_CORE` and `FIXED_CORE` modes. A Latin rotation changes
the execution order across rounds. One physical CPU is reserved for the heartbeat
process, and the workload CPU set excludes that core.

## Recorded evidence

The host trace adds thread CPU time, wall time minus thread CPU time, CPU IDs at
both event boundaries, context-switch deltas, and page-fault deltas. These fields
remain outside `transport_key`.

The heartbeat probe wakes every 100 microseconds by default. It writes a row only
when wakeup lateness reaches 50 microseconds, which limits observer overhead and
result size.

The analyzer builds robust event thresholds from `MULTI_CORE` samples using
`median + 3 * 1.4826 * MAD`. It then applies the same threshold to both binding
modes. This keeps the fixed-core comparison on one timing scale.

## Outputs

`runtime_stall_events.csv` contains the event-level join. The most useful columns
are `slow_event`, `heartbeat_or_preempt_signal`, `same_iteration_other_slow`,
`runtime_episode_signal`, and `wait_like_excess`.

`runtime_stall_summary.csv` reports association between slow events and runtime
signals. `SUPPORTS_RUNTIME_EPISODE` requires at least three slow events, at least
70 percent runtime-signal coverage among slow events, at most 20 percent among
normal events, and a risk ratio of at least three.

`runtime_stall_binding_comparison.csv` compares slow-event rates under the shared
multi-core threshold. `FIXED_CORE_REDUCES_SLOW_RATE` means the fixed-core rate is
at most half of the multi-core rate after at least three multi-core slow events.

`runtime_stall_thresholds.csv` records every baseline group and threshold, so the
classification remains auditable.

## Interpretation

Strong evidence for a runtime episode consists of `SUPPORTS_RUNTIME_EPISODE` in
the target row, supported by heartbeat or involuntary-switch evidence, or by a
slow neighboring SDK operation in the same iteration. A fixed-core rate reduction
adds evidence for host scheduling sensitivity.

An inconclusive result calls for more rounds with the same code and topology.
Recurring slow events with weak runtime association motivate a second-stage
label search. That second stage should use held-out traces before any field enters
the lookup key.

## Runner contract

`run_runtime_stall_probe.sh` requires a readable physical DPU topology TSV and a
NUMA-0 machine with enough selected ranks for 1024 DPUs. It builds separate 128
and 1024 DPU binaries, validates each trace, runs the analysis, and creates a
result archive by default.

Useful overrides are `TRACE_RUNS`, `PROCESS_WARMUP_RUNS`, `HOST_CPU_ID`,
`PROBE_CPU_ID`, and `MULTI_CORE_CPU_LIST`. Automatic CPU selection intersects
NUMA-local online CPUs with the caller's affinity mask and keeps one logical CPU
per physical core.
