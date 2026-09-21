# UPMEM SDK 2025.1.0 transfer trace patch

`upmem-2025.1.0-source.sha256` pins the clean source mirror at commit
`1647be840fb20074e7bc2cd6045b0d43ddfe5e51`. The public headers match the
installed UPMEM 2025.1.0 package used by the hardware host.

`upmem-2025.1.0-transfer-trace.patch` adds an in-memory trace to `libdpu`.
`UPMEM_TRANSFER_TRACE_CSV` enables collection, and
`UPMEM_TRANSFER_TRACE_RUN_ID` identifies the process. Records are written once
through an `atexit` handler. The fixed buffer holds 4096 records, and acceptance
requires `overflow_count=0`.

The patch records the public `dpu_push_xfer` interval as `PUSH_CALL`. It records
the hardware rank feature call as `BACKEND_TRANSFER`, with the end snapshot
taken after the backend returns. On Xeon SP this interval includes SDK worker
execution, AVX512 packing or unpacking, host memory traffic, PIM channel access,
and the backend fence.

The run script copies the source tree below `RESULT_ROOT`, verifies this hash
manifest, applies the patch to the copy, and builds only a replacement
`libdpu.so.2025.1`. The system `libdpuhw.so.2025.1` and kernel driver remain in
place.
