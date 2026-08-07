
#ifndef _MRAM_MANAGEMENT_H_
#define _MRAM_MANAGEMENT_H_

#include "../support/common.h"
#include "../support/utils.h"
#include "host_trace.h"

#define DPU_CAPACITY (64 << 20) // A DPU's capacity is 64 MiB

struct mram_heap_allocator_t {
    uint32_t totalAllocated;
};

static void init_allocator(struct mram_heap_allocator_t* allocator) {
    allocator->totalAllocated = 0;
}

static uint32_t mram_heap_alloc(struct mram_heap_allocator_t* allocator, uint32_t size) {
    uint32_t ret = allocator->totalAllocated;
    allocator->totalAllocated += ROUND_UP_TO_MULTIPLE_OF_8(size);
    if(allocator->totalAllocated > DPU_CAPACITY) {
        PRINT_ERROR("        Total memory allocated is %d bytes which exceeds the DPU capacity (%d bytes)!", allocator->totalAllocated, DPU_CAPACITY);
        exit(0);
    }
    return ret;
}

static void copyToDPUTraced(
    struct BfsHostTrace* trace,
    uint32_t globalDpuId,
    const char* subop,
    int32_t bfsLevel,
    struct dpu_set_t dpu,
    uint8_t* hostPtr,
    uint32_t mramIdx,
    uint32_t logicalSize
) {
    uint32_t transferSize = ROUND_UP_TO_MULTIPLE_OF_8(logicalSize);
    uint64_t startNs;
    uint64_t endNs;
    dpu_error_t status;

    if(bfsHostTraceEnabled(trace)) {
        startNs = bfsHostTraceNowNs();
    }
    status = dpu_copy_to(
        dpu, DPU_MRAM_HEAP_POINTER_NAME, mramIdx, hostPtr, transferSize
    );
    if(status != DPU_OK) {
        fprintf(stderr,
                "BFS_TRANSFER_ERROR direction=TO_DPU global_dpu_id=%u "
                "rank_ordinal=%u dpu_id_in_rank=%u subop=%s bfs_level=%d "
                "offset_bytes=%u logical_bytes=%u transfer_bytes=%u "
                "status=%s\n",
                globalDpuId, globalDpuId / 64, globalDpuId % 64,
                subop, bfsLevel, mramIdx, logicalSize, transferSize,
                dpu_error_to_string(status));
        fflush(stderr);
        DPU_ASSERT(status);
    }
    if(!bfsHostTraceEnabled(trace)) {
        return;
    }
    endNs = bfsHostTraceNowNs();
    bfsHostTraceRecord(trace, "dpu_copy_to", subop, bfsLevel, "TO_DPU", true,
                       globalDpuId, logicalSize, transferSize, mramIdx,
                       hostPtr,
                       startNs, endNs);
}

static void copyFromDPUTraced(
    struct BfsHostTrace* trace,
    uint32_t globalDpuId,
    const char* subop,
    int32_t bfsLevel,
    struct dpu_set_t dpu,
    uint32_t mramIdx,
    uint8_t* hostPtr,
    uint32_t logicalSize
) {
    uint32_t transferSize = ROUND_UP_TO_MULTIPLE_OF_8(logicalSize);
    uint64_t startNs;
    uint64_t endNs;
    dpu_error_t status;

    if(bfsHostTraceEnabled(trace)) {
        startNs = bfsHostTraceNowNs();
    }
    status = dpu_copy_from(
        dpu, DPU_MRAM_HEAP_POINTER_NAME, mramIdx, hostPtr, transferSize
    );
    if(status != DPU_OK) {
        fprintf(stderr,
                "BFS_TRANSFER_ERROR direction=FROM_DPU global_dpu_id=%u "
                "rank_ordinal=%u dpu_id_in_rank=%u subop=%s bfs_level=%d "
                "offset_bytes=%u logical_bytes=%u transfer_bytes=%u "
                "status=%s\n",
                globalDpuId, globalDpuId / 64, globalDpuId % 64,
                subop, bfsLevel, mramIdx, logicalSize, transferSize,
                dpu_error_to_string(status));
        fflush(stderr);
        DPU_ASSERT(status);
    }
    if(!bfsHostTraceEnabled(trace)) {
        return;
    }
    endNs = bfsHostTraceNowNs();
    bfsHostTraceRecord(trace, "dpu_copy_from", subop, bfsLevel, "FROM_DPU", true,
                       globalDpuId, logicalSize, transferSize, mramIdx,
                       hostPtr,
                       startNs, endNs);
}

#endif
