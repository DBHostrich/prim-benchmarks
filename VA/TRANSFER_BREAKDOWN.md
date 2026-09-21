# VA hardware transfer breakdown

This experiment measures the VA transfer path at the application boundary and
inside UPMEM SDK 2025.1.0. The application trace preserves the existing
`upmem.va_baseline.hw.v1` CSV and adds `upmem.va_transfer_breakdown.hw.v1`.
The patched `libdpu` emits `upmem.sdk_transfer_trace.v1`.

The script accepts these environment variables:

```text
RESULT_ROOT
SDK_SOURCE_ROOT
NUMA_NODE
VA_DPU_PROFILE
```

`RESULT_ROOT` uses a fresh path below `/tmp`. `SDK_SOURCE_ROOT` points at the
clean 2025.1.0 runtime source. The script verifies the pinned hashes, copies the
source below the result directory, applies the versioned patch, and builds a
shadow `libdpu.so.2025.1`. System `libdpuhw` and the kernel driver retain their
installed files.
The default profile is `backend=hw`, and the script requires that backend for
formal hardware acceptance. The shadow directory links the installed hardware
backend while replacing `libdpu.so.2025.1` with the instrumented build. It also
links the installed predefined programs at the relative path expected by
`libdpu`.

Run this command on the UPMEM hardware host from the repository root:

```bash
RESULT_ROOT="/tmp/va_transfer_breakdown_$(date +%Y%m%d_%H%M%S)" \
SDK_SOURCE_ROOT="/home/yiwei/bdang/mux-switch-oltpim/src/oltpim/oltpim-engine/upmem-sdk-runtime" \
NUMA_NODE=0 \
bash VA/run_hw_transfer_breakdown.sh
```

The workflow runs single-DPU and full-rank smoke checks, alternates stock and
instrumented overhead samples, executes five warmup processes, and collects
thirty full-rank processes. Each formal process carries `task-clock`, `cycles`,
and `instructions`. IMC collection records `COLLECTED` or a concrete access
status in `imc_status.txt`.

The summary manifest reports `functional_status` and `timing_status`
separately. `timing_status=REPORTED` requires p50 shifts within 3 percent and
p90 shifts within 5 percent for H2D and D2H.

## Coupled backend size sweep

The SDK hardware backend interleaves CPU loads, AVX512 transforms, accesses to
the memory-mapped PIM region, and fences inside the same worker loops. The size
sweep therefore treats `BACKEND_TRANSFER` as one coupled execution interval.
Process CPU time quantifies worker activity without converting it into an
additional wall-clock term.

Run the sweep on the hardware host:

```bash
RESULT_ROOT="/tmp/va_transfer_size_sweep_$(date +%Y%m%d_%H%M%S)" \
SDK_SOURCE_ROOT="/home/yiwei/bdang/mux-switch-oltpim/src/oltpim/oltpim-engine/upmem-sdk-runtime" \
NUMA_NODE=0 \
bash VA/run_hw_transfer_size_sweep.sh
```

The default sweep fixes `backend=hw,regionMode=perf` and uses these aggregate
input element counts:

```text
8192 16384 32768 131072 524288 1048576 2621440 4194304 8388608
```

They map to 512 B through 512 KiB per DPU on a 64-DPU rank and cover the SDK
worker thresholds. Each size has five warmup processes and thirty formal
processes. Formal samples alternate `AB` and `BA`, yielding fifteen samples per
order. Stock versus instrumented overhead is checked at the smallest, anchor,
and largest sizes.

The sweep adds these interfaces:

```text
VA_TRANSFER_COLLECTION_MODE
VA_SWEEP_INPUT_ELEMENTS
VA_SWEEP_OVERHEAD_ELEMENTS
VA_SWEEP_STRICT_OVERHEAD
VA_TRANSFER_ORDER
```

The default `VA_SWEEP_STRICT_OVERHEAD=0` records threshold excesses as
`timing_status=FAIL` while preserving `collection_status=PASS` and a successful
collection exit status. Set `VA_SWEEP_STRICT_OVERHEAD=1` when an overhead
threshold should produce a nonzero validation exit status. Manifest v2 records
both validator hashes so that a later revalidation remains attributable.

Revalidate an existing collection after updating the repository:

```bash
bash VA/revalidate_hw_transfer_size_sweep.sh \
  /tmp/va_transfer_size_sweep_<timestamp>
```

The command preserves the original summary and writes `summary_revalidated_v2`,
`validation_revalidated_v2.log`, revalidation provenance, and a
`.revalidated_v2.tar.gz` archive.

The output directory contains `collection_plan.csv` plus these summary files:

```text
summary/coupled_backend_samples.csv
summary/coupled_backend_summary.csv
summary/overhead_comparison.csv
summary/va_transfer_size_sweep_manifest.json
```

Copy the result from the local workstation after the hardware run:

```bash
mkdir -p /home/bdang/copy_result/2026_9_21
RESULT_NAME="va_transfer_size_sweep_<timestamp>"
rsync -avhP \
  "yiwei@feta.cs.northwestern.edu:/tmp/${RESULT_NAME}/" \
  "/home/bdang/copy_result/2026_9_21/${RESULT_NAME}/"
rsync -avhP \
  "yiwei@feta.cs.northwestern.edu:/tmp/${RESULT_NAME}.tar.gz" \
  "yiwei@feta.cs.northwestern.edu:/tmp/${RESULT_NAME}.tar.gz.sha256" \
  /home/bdang/copy_result/2026_9_21/
cd /home/bdang/copy_result/2026_9_21
sha256sum -c "${RESULT_NAME}.tar.gz.sha256"
```
