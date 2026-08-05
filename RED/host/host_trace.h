#ifndef _RED_HOST_TRACE_H_
#define _RED_HOST_TRACE_H_

#include <dpu.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct RedHostTraceEvent {
    uint64_t eventId;
    const char* op;
    const char* subop;
    const char* direction;
    int32_t iteration;
    int32_t warmup;
    uint32_t activeDpus;
    bool hasTransfer;
    uint64_t sizePerDpuBytes;
    uint64_t totalLogicalBytes;
    uint64_t totalTransferBytes;
    const char* targetSpace;
    const char* targetSymbol;
    uint64_t offsetBytes;
    uint64_t startNs;
    uint64_t endNs;
};

struct RedHostTraceDpu {
    uint64_t eventId;
    int32_t iteration;
    int32_t warmup;
    const char* op;
    const char* subop;
    const char* direction;
    uint32_t globalDpuId;
    uint64_t logicalBytes;
    uint64_t transferBytes;
};

struct RedHostTrace {
    bool enabled;
    const char* eventsOutputPath;
    const char* dpusOutputPath;
    const char* runId;
    uint64_t repeatId;
    uint32_t configuredDpus;
    uint32_t actualRanks;
    uint32_t numTasklets;
    uint64_t totalInputElements;
    uint64_t totalInputBytes;
    const char* hostNumaNode;
    const char* processState;
    uint64_t pretraceWarmupRuns;
    uint32_t* rankOrdinals;
    uint32_t* dpuIdsInRank;
    char* activeDpusPerRank;
    struct RedHostTraceEvent* events;
    size_t numEvents;
    size_t eventCapacity;
    struct RedHostTraceDpu* dpuRows;
    size_t numDpuRows;
    size_t dpuRowCapacity;
};

bool redHostTraceInit(
    struct RedHostTrace* trace,
    struct dpu_set_t dpuSet,
    uint32_t configuredDpus,
    uint32_t numTasklets,
    uint64_t totalInputElements,
    uint64_t totalInputBytes
);

bool redHostTraceRequested(void);
bool redHostTraceEnabled(const struct RedHostTrace* trace);
uint64_t redHostTraceNowNs(void);

void redHostTraceRecordEvent(
    struct RedHostTrace* trace,
    const char* op,
    const char* subop,
    int32_t iteration,
    int32_t warmup,
    uint32_t activeDpus,
    uint64_t startNs,
    uint64_t endNs
);

void redHostTraceRecordTransfer(
    struct RedHostTrace* trace,
    const char* subop,
    const char* direction,
    int32_t iteration,
    int32_t warmup,
    const char* targetSpace,
    const char* targetSymbol,
    uint64_t offsetBytes,
    const uint64_t* logicalBytesPerDpu,
    uint64_t uniformLogicalBytes,
    uint64_t sizePerDpuBytes,
    uint64_t startNs,
    uint64_t endNs
);

bool redHostTraceWrite(const struct RedHostTrace* trace);
void redHostTraceDestroy(struct RedHostTrace* trace);

#endif
