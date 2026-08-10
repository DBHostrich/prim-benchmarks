# GEMV controlled transfer-order probe

## Question

This experiment tests whether the variable state of the large `input_matrix`
transfer carries into the following 128-DPU `input_vector` transfer. It compares
two semantically equivalent orders before each DPU launch:

* `MATRIX_THEN_VECTOR` preserves the benchmark's original order.
* `VECTOR_THEN_MATRIX` moves the vector copy before the matrix copy.

The MRAM regions do not overlap, and both inputs reach the DPUs before launch.
The ordinary GEMV output equality check therefore remains the semantic gate.

## Scope and labels

The probe uses 128 DPUs, 16 tasklets, NUMA node 0, the same selected two ranks,
and one fixed host CPU. Each round contains both order variants, with their slot
order alternating across rounds.

`transfer_order_variant` is recorded as a diagnostic CSV column. It stays outside
the v9 `transport_key`. Existing predecessor fields follow the actual event order,
so `input_vector` records `input_matrix` as its predecessor in the original order
and `input_arguments` in the control order.

Observed predecessor duration is emitted only by the offline analysis. It is a
diagnostic correlation value and cannot enter a future lookup key.

## Heartbeat calibration

Before allocating DPUs, the runner records five seconds of idle heartbeat samples
at a one-microsecond logging threshold. The selected experiment threshold is:

```text
max(idle heartbeat P99, 200 microseconds)
```

`heartbeat_calibration_summary.csv` records the sample count, quantiles, maximum,
minimum threshold, and chosen threshold. Experiment heartbeat files contain only
samples crossing the calibrated threshold.

## Analysis outputs

`transfer_order_events.csv` contains every `input_vector` event, the current-order
slow threshold, exact-window heartbeat overlap, runtime counters, and the observed
duration of the immediately preceding SDK event.

`transfer_order_summary.csv` reports median, P10, P90, spread, CV, MAD, slow-event
rate, stability, and predecessor-duration correlations for each order and reuse
context.

`transfer_order_comparison.csv` applies each original-order threshold to both
orders. It also reports dispersion ratios and paired-round tail outcomes.

`SUPPORTS_PREDECESSOR_TRANSFER_STATE` requires at least 12 paired rounds, strong
dispersion reduction measured by spread and CV or by a large CV reduction, and at
least a twofold tail-rate reduction with three original-order slow events.
`NO_CLEAR_ORDER_EFFECT` means spread and CV both retain at least 80 percent of
their original-order values. Smaller collections report
`INSUFFICIENT_PAIRED_ROUNDS`; other outcomes report `ORDER_EFFECT_INCONCLUSIVE`.

## Collection sizes

A gate uses three traced processes per order. It validates compilation, output
equality, trace sequence, physical topology, heartbeat calibration, and analysis
generation. Its statistical decision commonly remains sample-limited.

The formal default uses 24 traced processes per order. Each process contributes
one `FIRST_USE` and three `REUSED` input-vector events, giving 24 first-use and 72
reused events per order.
