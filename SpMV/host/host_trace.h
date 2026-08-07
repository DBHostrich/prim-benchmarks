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
    uintptr_t hostBufferAddress;
    uint64_t opCallIndex;
    bool hasDpuOpCallIndex;
    uint64_t dpuOpCallIndex;
    uint64_t startNs;
    uint64_t endNs;
};

#define SPMV_HOST_TRACE_OP_SLOTS 7

struct SpmvHostTrace {
    bool enabled;
    const char* outputPath;
    const char* runId;
    uint64_t repeatId;
    uint32_t configuredDpus;
    uint32_t actualRanks;
    uint32_t numTasklets;
    const char* hostNumaNode;
    const char* processState;
    uint64_t pretraceWarmupRuns;
    uint32_t* rankOrdinals;
    uint32_t* dpuIdsInRank;
    uint32_t* sdkPhysicalRankIds;
    uint32_t* dpuSysfsRankIds;
    uint32_t* dpuRankNumaNodes;
    uint32_t* dpuChannelIds;
    uint32_t* sdkSliceIds;
    uint32_t* sdkMemberIds;
    uint64_t* dpuCopyToCallCounts;
    uint64_t* dpuCopyFromCallCounts;
    uint64_t opCallCounts[SPMV_HOST_TRACE_OP_SLOTS];
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

bool spmvHostTraceRequested(void);
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
    const void* hostBuffer,
    uint64_t startNs,
    uint64_t endNs
);

bool spmvHostTraceWrite(const struct SpmvHostTrace* trace);
void spmvHostTraceDestroy(struct SpmvHostTrace* trace);

#endif
