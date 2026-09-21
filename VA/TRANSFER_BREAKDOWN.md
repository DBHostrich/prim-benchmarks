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

Copy the result from the local workstation after the hardware run:

```bash
mkdir -p /home/bdang/copy_result/2026_9_21
rsync -avhP \
  yiwei@feta.cs.northwestern.edu:/tmp/va_transfer_breakdown_<timestamp>/ \
  /home/bdang/copy_result/2026_9_21/va_transfer_breakdown_<timestamp>/
rsync -avhP \
  yiwei@feta.cs.northwestern.edu:/tmp/va_transfer_breakdown_<timestamp>.tar.gz* \
  /home/bdang/copy_result/2026_9_21/
cd /home/bdang/copy_result/2026_9_21
sha256sum -c va_transfer_breakdown_<timestamp>.tar.gz.sha256
```
