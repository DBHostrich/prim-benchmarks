#ifndef _SPMV_HOST_TRACE_H_
#define _SPMV_HOST_TRACE_H_

#include <dpu.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct SpmvHostTraceEvent {
    uint64_t eventId;
    const char* op;
    const char* subop;
    const char* direction;
    bool hasDpu;
    uint32_t globalDpuId;
    uint64_t logicalBytes;
    uint64_t transferBytes;
    uint64_t offsetBytes;
    uint64_t startNs;
    uint64_t endNs;
};

struct SpmvHostTrace {
    bool enabled;
    const char* outputPath;
    const char* runId;
    uint64_t repeatId;
    uint32_t configuredDpus;
    uint32_t actualRanks;
    uint32_t numTasklets;
    uint32_t* rankOrdinals;
    uint32_t* dpuIdsInRank;
    struct SpmvHostTraceEvent* events;
    size_t numEvents;
    size_t capacity;
};

bool spmvHostTraceInit(
    struct SpmvHostTrace* trace,
    struct dpu_set_t dpuSet,
    uint32_t configuredDpus,
    uint32_t numTasklets
);

bool spmvHostTraceEnabled(const struct SpmvHostTrace* trace);
uint64_t spmvHostTraceNowNs(void);

void spmvHostTraceRecord(
    struct SpmvHostTrace* trace,
    const char* op,
    const char* subop,
    const char* direction,
    bool hasDpu,
    uint32_t globalDpuId,
    uint64_t logicalBytes,
    uint64_t transferBytes,
    uint64_t offsetBytes,
    uint64_t startNs,
    uint64_t endNs
);

bool spmvHostTraceWrite(const struct SpmvHostTrace* trace);
void spmvHostTraceDestroy(struct SpmvHostTrace* trace);

#endif
