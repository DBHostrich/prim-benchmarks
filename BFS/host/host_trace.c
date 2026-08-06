#define _GNU_SOURCE

#include "host_trace.h"

#include <dpu_management.h>

#include <errno.h>
#include <inttypes.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static bool parseUnsignedEnv(const char* name, const char* value, uint64_t* parsedValue) {
    char* end = NULL;
    unsigned long long parsed;

    if(value == NULL || value[0] == '\0') {
        *parsedValue = 0;
        return true;
    }

    errno = 0;
    parsed = strtoull(value, &end, 10);
    if(errno != 0 || end == value || *end != '\0') {
        fprintf(stderr, "Invalid %s: %s\n", name, value);
        return false;
    }
    *parsedValue = (uint64_t)parsed;
    return true;
}

static bool isLabelAtom(const char* value) {
    const unsigned char* p = (const unsigned char*)value;
    if(value == NULL || value[0] == '\0') {
        return false;
    }
    for(; *p != '\0'; ++p) {
        if(!isalnum(*p) && *p != '_' && *p != '-' && *p != '.') {
            return false;
        }
    }
    return true;
}

static size_t opSlot(const char* op) {
    if(strcmp(op, "dpu_alloc") == 0) {
        return 0;
    }
    if(strcmp(op, "dpu_load") == 0) {
        return 1;
    }
    if(strcmp(op, "dpu_copy_to") == 0) {
        return 2;
    }
    if(strcmp(op, "dpu_copy_from") == 0) {
        return 3;
    }
    if(strcmp(op, "dpu_launch") == 0) {
        return 4;
    }
    if(strcmp(op, "dpu_free") == 0) {
        return 5;
    }
    return 6;
}

static bool isTransferEvent(const struct BfsHostTraceEvent* event) {
    return strcmp(event->op, "dpu_copy_to") == 0
        || strcmp(event->op, "dpu_copy_from") == 0;
}

static const char* sdkApiKind(const struct BfsHostTraceEvent* event) {
    if(strcmp(event->op, "dpu_copy_to") == 0
       || strcmp(event->op, "dpu_copy_from") == 0) {
        return "SINGLE_COPY";
    }
    return "";
}

static const char* logicalDistributionClass(
    const struct BfsHostTraceEvent* event
) {
    if(strcmp(event->op, "dpu_copy_to") == 0) {
        if(strcmp(event->subop, "visited_init") == 0
           || strcmp(event->subop, "frontier_init") == 0
           || strcmp(event->subop, "frontier_broadcast") == 0) {
            return "SHARED_REPLICATION";
        }
        return "PARTITIONED_SCATTER";
    }
    if(strcmp(event->op, "dpu_copy_from") == 0) {
        /* Frontier bitmaps overlap logically and are OR-reduced by the host. */
        if(strcmp(event->subop, "frontier_result") == 0) {
            return "REDUCTION_GATHER";
        }
        return "PARTITIONED_GATHER";
    }
    return "";
}

static const char* sameSourceAcrossGroup(
    const struct BfsHostTraceEvent* event
) {
    if(strcmp(event->op, "dpu_copy_to") == 0) {
        if(strcmp(event->subop, "visited_init") == 0
           || strcmp(event->subop, "frontier_init") == 0
           || strcmp(event->subop, "frontier_broadcast") == 0) {
            return "1";
        }
        return "0";
    }
    if(strcmp(event->op, "dpu_copy_from") == 0) {
        return "0";
    }
    return "";
}

static const char* phaseClass(const struct BfsHostTraceEvent* event) {
    if(!isTransferEvent(event)) {
        return "";
    }
    if(strcmp(event->subop, "node_ptrs") == 0
       || strcmp(event->subop, "neighbor_idxs") == 0
       || strcmp(event->subop, "node_level_init") == 0
       || strcmp(event->subop, "visited_init") == 0
       || strcmp(event->subop, "frontier_init") == 0
       || strcmp(event->subop, "params_init") == 0) {
        return "INIT";
    }
    if(strcmp(event->subop, "frontier_result") == 0
       || strcmp(event->subop, "frontier_broadcast") == 0
       || strcmp(event->subop, "params_level") == 0) {
        return "ITERATIVE";
    }
    if(strcmp(event->subop, "node_level_result") == 0) {
        return "FINALIZE";
    }
    return "UNKNOWN";
}

struct BfsDerivedTransferContext {
    const char* previousSdkOp;
    const char* previousSdkDirection;
    uint64_t previousSdkTransferBytes;
    const char* previousSdkTopologyRelation;
    uint64_t nsSincePreviousSdkEvent;
    const char* previousDpuDirection;
    uint64_t previousDpuTransferBytes;
    const char* previousDpuTargetRelation;
    uint64_t launchesSincePreviousDpuTransfer;
    const char* targetRegionReuseClass;
    uint64_t hostBufferPageOffset;
    const char* hostBufferReuseClass;
};

struct BfsHostBufferState {
    uintptr_t address;
    uint64_t transferBytes;
    const char* direction;
};

static const char* sdkTopologyRelation(
    const struct BfsHostTrace* trace,
    const struct BfsHostTraceEvent* previous,
    const struct BfsHostTraceEvent* current
) {
    if(previous == NULL) {
        return "NONE";
    }
    if(!previous->hasDpu) {
        return "COLLECTION";
    }
    if(previous->globalDpuId == current->globalDpuId) {
        return "SAME_DPU";
    }
    if(trace->sdkPhysicalRankIds[previous->globalDpuId]
       != trace->sdkPhysicalRankIds[current->globalDpuId]) {
        return "OTHER_RANK";
    }
    if(trace->sdkSliceIds[previous->globalDpuId]
       != trace->sdkSliceIds[current->globalDpuId]) {
        return "SAME_RANK";
    }
    if(trace->sdkMemberIds[previous->globalDpuId] / 2
       == trace->sdkMemberIds[current->globalDpuId] / 2) {
        return "SAME_MUX_PAIR";
    }
    return "SAME_SLICE";
}

static bool deriveTransferContexts(
    const struct BfsHostTrace* trace,
    struct BfsDerivedTransferContext* contexts
) {
    size_t* previousDpuEvent;
    size_t* lastDpuEvent;
    uint64_t* lastDpuTransferLaunchCount;
    struct BfsHostBufferState* hostBuffers;
    size_t hostBufferCount = 0;
    uint64_t launchCount = 0;
    size_t i;

    previousDpuEvent = malloc(trace->numEvents * sizeof(*previousDpuEvent));
    lastDpuEvent = malloc(trace->configuredDpus * sizeof(*lastDpuEvent));
    lastDpuTransferLaunchCount = calloc(
        trace->configuredDpus, sizeof(*lastDpuTransferLaunchCount)
    );
    hostBuffers = malloc(trace->numEvents * sizeof(*hostBuffers));
    if(previousDpuEvent == NULL || lastDpuEvent == NULL
       || lastDpuTransferLaunchCount == NULL || hostBuffers == NULL) {
        free(previousDpuEvent);
        free(lastDpuEvent);
        free(lastDpuTransferLaunchCount);
        free(hostBuffers);
        return false;
    }
    for(i = 0; i < trace->configuredDpus; ++i) {
        lastDpuEvent[i] = SIZE_MAX;
    }

    for(i = 0; i < trace->numEvents; ++i) {
        const struct BfsHostTraceEvent* event = &trace->events[i];
        struct BfsDerivedTransferContext* context = &contexts[i];
        const struct BfsHostTraceEvent* previousSdk = i == 0
            ? NULL : &trace->events[i - 1];
        size_t previousIndex;
        size_t cursor;
        size_t hostIndex;
        bool targetSeen = false;

        previousDpuEvent[i] = SIZE_MAX;
        if(strcmp(event->op, "dpu_launch") == 0) {
            ++launchCount;
        }
        if(!isTransferEvent(event)) {
            continue;
        }

        context->previousSdkOp = previousSdk == NULL
            ? "NONE" : previousSdk->op;
        context->previousSdkDirection = previousSdk == NULL
            || previousSdk->direction[0] == '\0'
            ? "NONE" : previousSdk->direction;
        context->previousSdkTransferBytes = previousSdk != NULL
            && isTransferEvent(previousSdk)
            ? previousSdk->transferBytes : 0;
        context->previousSdkTopologyRelation = sdkTopologyRelation(
            trace, previousSdk, event
        );
        context->nsSincePreviousSdkEvent = previousSdk == NULL
            || event->startNs < previousSdk->endNs
            ? 0 : event->startNs - previousSdk->endNs;

        previousIndex = lastDpuEvent[event->globalDpuId];
        previousDpuEvent[i] = previousIndex;
        if(previousIndex == SIZE_MAX) {
            context->previousDpuDirection = "NONE";
            context->previousDpuTransferBytes = 0;
            context->previousDpuTargetRelation = "NONE";
        } else {
            const struct BfsHostTraceEvent* previousDpu =
                &trace->events[previousIndex];
            context->previousDpuDirection = previousDpu->direction;
            context->previousDpuTransferBytes = previousDpu->transferBytes;
            context->previousDpuTargetRelation =
                previousDpu->offsetBytes == event->offsetBytes
                && previousDpu->transferBytes == event->transferBytes
                ? "SAME_REGION" : "DIFFERENT_REGION";
        }
        context->launchesSincePreviousDpuTransfer =
            launchCount - lastDpuTransferLaunchCount[event->globalDpuId];

        cursor = previousIndex;
        while(cursor != SIZE_MAX) {
            const struct BfsHostTraceEvent* prior = &trace->events[cursor];
            if(prior->offsetBytes == event->offsetBytes
               && prior->transferBytes == event->transferBytes) {
                targetSeen = true;
                break;
            }
            cursor = previousDpuEvent[cursor];
        }
        context->targetRegionReuseClass = targetSeen
            ? "REUSED_REGION" : "FIRST_REGION_ACCESS";

        context->hostBufferPageOffset =
            (uint64_t)(event->hostBufferAddress % (uintptr_t)4096);
        context->hostBufferReuseClass = "FIRST_SDK_USE";
        for(hostIndex = 0; hostIndex < hostBufferCount; ++hostIndex) {
            if(hostBuffers[hostIndex].address == event->hostBufferAddress
               && hostBuffers[hostIndex].transferBytes == event->transferBytes) {
                context->hostBufferReuseClass =
                    strcmp(hostBuffers[hostIndex].direction, event->direction) == 0
                    ? "SAME_DIRECTION_REUSE" : "DIRECTION_SWITCH_REUSE";
                hostBuffers[hostIndex].direction = event->direction;
                break;
            }
        }
        if(hostIndex == hostBufferCount) {
            hostBuffers[hostBufferCount].address = event->hostBufferAddress;
            hostBuffers[hostBufferCount].transferBytes = event->transferBytes;
            hostBuffers[hostBufferCount].direction = event->direction;
            ++hostBufferCount;
        }

        lastDpuEvent[event->globalDpuId] = i;
        lastDpuTransferLaunchCount[event->globalDpuId] = launchCount;
    }

    free(previousDpuEvent);
    free(lastDpuEvent);
    free(lastDpuTransferLaunchCount);
    free(hostBuffers);
    return true;
}

static void formatOffsetFeature(
    const struct BfsHostTraceEvent* event,
    char* output,
    size_t outputSize
) {
    uint64_t page;
    uint64_t pages;

    if(!event->hasDpu) {
        snprintf(output, outputSize, "none");
        return;
    }
    page = event->offsetBytes / UINT64_C(4096);
    pages = event->transferBytes == 0 ? 0
        : ((event->offsetBytes % UINT64_C(4096)) + event->transferBytes
           + UINT64_C(4095)) / UINT64_C(4096);
    snprintf(output, outputSize,
             "off=%" PRIu64 ":a8=%u:a64=%u:p4k=%" PRIu64 ":pages=%" PRIu64,
             event->offsetBytes, event->offsetBytes % 8 == 0,
             event->offsetBytes % 64 == 0, page, pages);
}

static void formatCallContext(
    const struct BfsHostTrace* trace,
    const struct BfsHostTraceEvent* event,
    char* output,
    size_t outputSize
) {
    if(event->hasDpuOpCallIndex) {
        snprintf(output, outputSize,
                 "opidx=%" PRIu64 ":dpuopidx=%" PRIu64
                 ":process=%s:prewarm=%" PRIu64,
                 event->opCallIndex, event->dpuOpCallIndex,
                 trace->processState, trace->pretraceWarmupRuns);
    } else {
        snprintf(output, outputSize,
                 "opidx=%" PRIu64 ":dpuopidx=none"
                 ":process=%s:prewarm=%" PRIu64,
                 event->opCallIndex, trace->processState,
                 trace->pretraceWarmupRuns);
    }
}

static bool formatTransportKey(
    const struct BfsHostTrace* trace,
    const struct BfsHostTraceEvent* event,
    const struct BfsDerivedTransferContext* context,
    const char* eventSdkApiKind,
    const char* distributionClass,
    const char* sameSource,
    char* output,
    size_t outputSize
) {
    int result;
    if(!event->hasDpu || !isTransferEvent(event)) {
        output[0] = '\0';
        return true;
    }

    /* Every BFS copy wrapper receives one DPU selected by DPU_FOREACH. */
    result = snprintf(
        output, outputSize,
        "v6;op=%s;direction=%s;sdk_api_kind=%s;"
        "logical_distribution_class=%s;target_space=MRAM;"
        "transfer_bytes_per_dpu=%" PRIu64
        ";active_dpus=1;active_ranks=1;active_dpus_per_rank=1;"
        "rank_ordinal=%u;dpu_id_in_rank=%u;same_source_across_group=%s;"
        "sdk_physical_rank_id=%u;sdk_slice_id=%u;sdk_member_id=%u;"
        "previous_sdk_topology_relation=%s;"
        "previous_dpu_direction=%s;previous_dpu_target_relation=%s;"
        "target_region_reuse_class=%s;host_numa_node=%s",
        event->op, event->direction, eventSdkApiKind, distributionClass,
        event->transferBytes, trace->rankOrdinals[event->globalDpuId],
        trace->dpuIdsInRank[event->globalDpuId], sameSource,
        trace->sdkPhysicalRankIds[event->globalDpuId],
        trace->sdkSliceIds[event->globalDpuId],
        trace->sdkMemberIds[event->globalDpuId],
        context->previousSdkTopologyRelation,
        context->previousDpuDirection, context->previousDpuTargetRelation,
        context->targetRegionReuseClass, trace->hostNumaNode
    );
    return result >= 0 && (size_t)result < outputSize;
}

static void writeCsvString(FILE* fp, const char* value) {
    const char* p;
    bool quote = false;

    if(value == NULL) {
        return;
    }
    for(p = value; *p != '\0'; ++p) {
        if(*p == ',' || *p == '"' || *p == '\n' || *p == '\r') {
            quote = true;
            break;
        }
    }
    if(!quote) {
        fputs(value, fp);
        return;
    }

    fputc('"', fp);
    for(p = value; *p != '\0'; ++p) {
        if(*p == '"') {
            fputc('"', fp);
        }
        fputc(*p, fp);
    }
    fputc('"', fp);
}

static void growEvents(struct BfsHostTrace* trace) {
    size_t newCapacity;
    struct BfsHostTraceEvent* newEvents;

    if(trace->capacity > SIZE_MAX / (2u * sizeof(*trace->events))) {
        fprintf(stderr, "BFS host trace capacity overflow\n");
        exit(EXIT_FAILURE);
    }
    newCapacity = trace->capacity == 0 ? 64u : trace->capacity * 2u;
    newEvents = realloc(trace->events, newCapacity * sizeof(*trace->events));
    if(newEvents == NULL) {
        fprintf(stderr, "Could not grow the BFS host trace buffer to %zu events\n",
                newCapacity);
        exit(EXIT_FAILURE);
    }
    memset(newEvents + trace->capacity, 0,
           (newCapacity - trace->capacity) * sizeof(*newEvents));
    trace->events = newEvents;
    trace->capacity = newCapacity;
}

bool bfsHostTraceRequested(void) {
    const char* outputPath = getenv("BFS_TRACE_CSV");
    return outputPath != NULL && outputPath[0] != '\0';
}

bool bfsHostTraceInit(
    struct BfsHostTrace* trace,
    struct dpu_set_t dpuSet,
    uint32_t configuredDpus,
    uint32_t numTasklets
) {
    const char* outputPath = getenv("BFS_TRACE_CSV");
    const char* runId = getenv("BFS_TRACE_RUN_ID");
    const char* repeatId = getenv("BFS_TRACE_REPEAT_ID");
    const char* hostNumaNode = getenv("BFS_TRACE_HOST_NUMA_NODE");
    const char* processState = getenv("BFS_TRACE_PROCESS_STATE");
    const char* pretraceWarmupRuns = getenv("BFS_TRACE_PREWARM_RUNS");
    struct dpu_set_t rank = {0};
    struct dpu_set_t dpu = {0};
    struct dpu_rank_t* sdkRank;
    uint32_t rankOrdinal;
    uint32_t dpuIdInRank;
    uint32_t globalDpuId = 0;

    memset(trace, 0, sizeof(*trace));
    if(!bfsHostTraceRequested()) {
        return true;
    }

    trace->outputPath = outputPath;
    trace->runId = (runId == NULL || runId[0] == '\0') ? "bfs" : runId;
    trace->configuredDpus = configuredDpus;
    trace->numTasklets = numTasklets;
    trace->hostNumaNode = (hostNumaNode == NULL || hostNumaNode[0] == '\0')
        ? "unknown" : hostNumaNode;
    trace->processState = (processState == NULL || processState[0] == '\0')
        ? "fresh_process" : processState;
    trace->capacity = (size_t)configuredDpus * 16u + 64u;
    if(!parseUnsignedEnv("BFS_TRACE_REPEAT_ID", repeatId, &trace->repeatId)
       || !parseUnsignedEnv("BFS_TRACE_PREWARM_RUNS", pretraceWarmupRuns,
                            &trace->pretraceWarmupRuns)) {
        return false;
    }
    if(!isLabelAtom(trace->hostNumaNode) || !isLabelAtom(trace->processState)) {
        fprintf(stderr, "BFS trace NUMA/process label contains unsupported characters\n");
        return false;
    }
    if(dpu_get_nr_ranks(dpuSet, &trace->actualRanks) != DPU_OK) {
        fprintf(stderr, "Could not query the allocated rank count\n");
        return false;
    }

    trace->rankOrdinals = calloc(configuredDpus, sizeof(*trace->rankOrdinals));
    trace->dpuIdsInRank = calloc(configuredDpus, sizeof(*trace->dpuIdsInRank));
    trace->sdkPhysicalRankIds = calloc(
        configuredDpus, sizeof(*trace->sdkPhysicalRankIds)
    );
    trace->sdkSliceIds = calloc(configuredDpus, sizeof(*trace->sdkSliceIds));
    trace->sdkMemberIds = calloc(configuredDpus, sizeof(*trace->sdkMemberIds));
    trace->dpuCopyToCallCounts = calloc(configuredDpus,
                                        sizeof(*trace->dpuCopyToCallCounts));
    trace->dpuCopyFromCallCounts = calloc(configuredDpus,
                                          sizeof(*trace->dpuCopyFromCallCounts));
    trace->events = calloc(trace->capacity, sizeof(*trace->events));
    if(trace->rankOrdinals == NULL || trace->dpuIdsInRank == NULL
       || trace->sdkPhysicalRankIds == NULL
       || trace->sdkSliceIds == NULL || trace->sdkMemberIds == NULL
       || trace->dpuCopyToCallCounts == NULL
       || trace->dpuCopyFromCallCounts == NULL || trace->events == NULL) {
        fprintf(stderr, "Could not allocate the BFS host trace buffers\n");
        bfsHostTraceDestroy(trace);
        return false;
    }

    DPU_RANK_FOREACH(dpuSet, rank, rankOrdinal) {
        sdkRank = dpu_rank_from_set(rank);
        if(sdkRank == NULL) {
            fprintf(stderr, "Could not resolve SDK rank %u\n", rankOrdinal);
            bfsHostTraceDestroy(trace);
            return false;
        }
        DPU_FOREACH(rank, dpu, dpuIdInRank) {
            if(globalDpuId >= configuredDpus) {
                fprintf(stderr, "Allocated DPU topology exceeds configured DPU count\n");
                bfsHostTraceDestroy(trace);
                return false;
            }
            trace->rankOrdinals[globalDpuId] = rankOrdinal;
            trace->dpuIdsInRank[globalDpuId] = dpuIdInRank;
            trace->sdkPhysicalRankIds[globalDpuId] = dpu_get_rank_id(sdkRank);
            trace->sdkSliceIds[globalDpuId] = dpu_get_slice_id(dpu.dpu);
            trace->sdkMemberIds[globalDpuId] = dpu_get_member_id(dpu.dpu);
            ++globalDpuId;
        }
    }
    if(globalDpuId != configuredDpus) {
        fprintf(stderr, "Allocated DPU topology contains %u DPUs, expected %u\n",
                globalDpuId, configuredDpus);
        bfsHostTraceDestroy(trace);
        return false;
    }

    trace->enabled = true;
    return true;
}

bool bfsHostTraceEnabled(const struct BfsHostTrace* trace) {
    return trace != NULL && trace->enabled;
}

uint64_t bfsHostTraceNowNs(void) {
    struct timespec now;
    if(clock_gettime(CLOCK_MONOTONIC_RAW, &now) != 0) {
        perror("clock_gettime(CLOCK_MONOTONIC_RAW)");
        exit(EXIT_FAILURE);
    }
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
}

void bfsHostTraceRecord(
    struct BfsHostTrace* trace,
    const char* op,
    const char* subop,
    int32_t bfsLevel,
    const char* direction,
    bool hasDpu,
    uint32_t globalDpuId,
    uint64_t logicalBytes,
    uint64_t transferBytes,
    uint64_t offsetBytes,
    const void* hostBuffer,
    uint64_t startNs,
    uint64_t endNs
) {
    struct BfsHostTraceEvent* event;

    if(!bfsHostTraceEnabled(trace)) {
        return;
    }
    if(trace->numEvents >= trace->capacity) {
        growEvents(trace);
    }
    if(hasDpu && globalDpuId >= trace->configuredDpus) {
        fprintf(stderr, "BFS host trace DPU ID %u is out of range\n", globalDpuId);
        exit(EXIT_FAILURE);
    }
    if(endNs < startNs) {
        fprintf(stderr, "BFS host trace clock moved backwards\n");
        exit(EXIT_FAILURE);
    }

    event = &trace->events[trace->numEvents];
    event->eventId = trace->numEvents;
    event->op = op;
    event->subop = subop;
    event->bfsLevel = bfsLevel;
    event->direction = direction;
    event->hasDpu = hasDpu;
    event->globalDpuId = globalDpuId;
    event->logicalBytes = logicalBytes;
    event->transferBytes = transferBytes;
    event->offsetBytes = offsetBytes;
    event->hostBufferAddress = (uintptr_t)hostBuffer;
    event->opCallIndex = trace->opCallCounts[opSlot(op)]++;
    if(hasDpu && strcmp(op, "dpu_copy_to") == 0) {
        event->hasDpuOpCallIndex = true;
        event->dpuOpCallIndex = trace->dpuCopyToCallCounts[globalDpuId]++;
    } else if(hasDpu && strcmp(op, "dpu_copy_from") == 0) {
        event->hasDpuOpCallIndex = true;
        event->dpuOpCallIndex = trace->dpuCopyFromCallCounts[globalDpuId]++;
    }
    event->startNs = startNs;
    event->endNs = endNs;
    ++trace->numEvents;
}

bool bfsHostTraceWrite(const struct BfsHostTrace* trace) {
    FILE* fp;
    struct BfsDerivedTransferContext* contexts;
    size_t i;

    if(!bfsHostTraceEnabled(trace)) {
        return true;
    }
    fp = fopen(trace->outputPath, "w");
    if(fp == NULL) {
        fprintf(stderr, "Could not open BFS trace %s: %s\n",
                trace->outputPath, strerror(errno));
        return false;
    }

    contexts = calloc(trace->numEvents, sizeof(*contexts));
    if(contexts == NULL || !deriveTransferContexts(trace, contexts)) {
        fprintf(stderr, "Could not derive BFS transfer hardware contexts\n");
        free(contexts);
        fclose(fp);
        return false;
    }

    fputs("run_id,repeat_id,event_id,configured_dpus,actual_ranks,num_tasklets,"
          "op,direction,sdk_api_kind,logical_distribution_class,target_space,"
          "transfer_bytes_per_dpu,active_dpus,active_ranks,active_dpus_per_rank,"
          "rank_ordinal,dpu_id_in_rank,sdk_physical_rank_id,"
          "sdk_slice_id,sdk_member_id,physical_dpu_identity,"
          "same_source_across_group,phase_class,"
          "subop,bfs_level,global_dpu_id,target_symbol,offset_bytes,offset_feature,"
          "logical_bytes,transfer_bytes,host_buffer_address,"
          "host_buffer_page_offset,host_buffer_reuse_class,"
          "previous_sdk_op,previous_sdk_direction,previous_sdk_transfer_bytes,"
          "previous_sdk_topology_relation,ns_since_previous_sdk_event,"
          "previous_dpu_direction,previous_dpu_transfer_bytes,"
          "previous_dpu_target_relation,launches_since_previous_dpu_transfer,"
          "target_region_reuse_class,"
          "op_call_index,dpu_op_call_index,process_state,pretrace_warmup_runs,"
          "host_numa_node,call_context,transport_key,"
          "host_start_ns,host_end_ns,measured_ns\n", fp);

    for(i = 0; i < trace->numEvents; ++i) {
        const struct BfsHostTraceEvent* event = &trace->events[i];
        const struct BfsDerivedTransferContext* context = &contexts[i];
        const char* eventSdkApiKind = sdkApiKind(event);
        const char* distributionClass = logicalDistributionClass(event);
        const char* sameSource = sameSourceAcrossGroup(event);
        const char* eventPhaseClass = phaseClass(event);
        char offsetFeature[160];
        char callContext[256];
        char physicalDpuIdentity[128];
        char transportKey[1024];

        formatOffsetFeature(event, offsetFeature, sizeof(offsetFeature));
        formatCallContext(trace, event, callContext, sizeof(callContext));
        if(event->hasDpu) {
            snprintf(
                physicalDpuIdentity, sizeof(physicalDpuIdentity),
                "rank:%u/slice:%u/member:%u",
                trace->sdkPhysicalRankIds[event->globalDpuId],
                trace->sdkSliceIds[event->globalDpuId],
                trace->sdkMemberIds[event->globalDpuId]
            );
        } else {
            physicalDpuIdentity[0] = '\0';
        }
        if(!formatTransportKey(trace, event, context, eventSdkApiKind,
                              distributionClass, sameSource,
                              transportKey,
                              sizeof(transportKey))) {
            fprintf(stderr, "BFS transport key is too long for event %" PRIu64 "\n",
                    event->eventId);
            free(contexts);
            fclose(fp);
            return false;
        }

        writeCsvString(fp, trace->runId);
        fprintf(fp, ",%" PRIu64 ",%" PRIu64 ",%u,%u,%u,",
                trace->repeatId, event->eventId, trace->configuredDpus,
                trace->actualRanks, trace->numTasklets);
        writeCsvString(fp, event->op);
        fputc(',', fp);
        writeCsvString(fp, event->direction);
        fputc(',', fp);
        writeCsvString(fp, eventSdkApiKind);
        fputc(',', fp);
        writeCsvString(fp, distributionClass);
        fputc(',', fp);
        if(event->hasDpu) {
            fprintf(fp, "MRAM,%" PRIu64 ",1,1,1,%u,%u,%u,%u,%u,",
                    event->transferBytes,
                    trace->rankOrdinals[event->globalDpuId],
                    trace->dpuIdsInRank[event->globalDpuId],
                    trace->sdkPhysicalRankIds[event->globalDpuId],
                    trace->sdkSliceIds[event->globalDpuId],
                    trace->sdkMemberIds[event->globalDpuId]);
            writeCsvString(fp, physicalDpuIdentity);
            fputc(',', fp);
            writeCsvString(fp, sameSource);
        } else {
            fputs(",,,,,,,,,,,", fp);
        }
        fputc(',', fp);
        writeCsvString(fp, eventPhaseClass);
        fputc(',', fp);
        writeCsvString(fp, event->subop);
        fputc(',', fp);
        if(event->bfsLevel >= 0) {
            fprintf(fp, "%" PRId32, event->bfsLevel);
        }
        fputc(',', fp);
        if(event->hasDpu) {
            fprintf(fp, "%u,DPU_MRAM_HEAP_POINTER_NAME,", event->globalDpuId);
            fprintf(fp, "%" PRIu64 ",", event->offsetBytes);
            writeCsvString(fp, offsetFeature);
            fprintf(fp, ",%" PRIu64 ",%" PRIu64 ",%" PRIuPTR
                    ",%" PRIu64 ",",
                    event->logicalBytes, event->transferBytes,
                    event->hostBufferAddress, context->hostBufferPageOffset);
            writeCsvString(fp, context->hostBufferReuseClass);
            fputc(',', fp);
            writeCsvString(fp, context->previousSdkOp);
            fputc(',', fp);
            writeCsvString(fp, context->previousSdkDirection);
            fprintf(fp, ",%" PRIu64 ",",
                    context->previousSdkTransferBytes);
            writeCsvString(fp, context->previousSdkTopologyRelation);
            fprintf(fp, ",%" PRIu64 ",",
                    context->nsSincePreviousSdkEvent);
            writeCsvString(fp, context->previousDpuDirection);
            fprintf(fp, ",%" PRIu64 ",",
                    context->previousDpuTransferBytes);
            writeCsvString(fp, context->previousDpuTargetRelation);
            fprintf(fp, ",%" PRIu64 ",",
                    context->launchesSincePreviousDpuTransfer);
            writeCsvString(fp, context->targetRegionReuseClass);
        } else {
            size_t emptyField;
            fputs(",,,none", fp);
            for(emptyField = 0; emptyField < 15; ++emptyField) {
                fputc(',', fp);
            }
        }
        fprintf(fp, ",%" PRIu64 ",", event->opCallIndex);
        if(event->hasDpuOpCallIndex) {
            fprintf(fp, "%" PRIu64, event->dpuOpCallIndex);
        }
        fputc(',', fp);
        writeCsvString(fp, trace->processState);
        fprintf(fp, ",%" PRIu64 ",", trace->pretraceWarmupRuns);
        writeCsvString(fp, trace->hostNumaNode);
        fputc(',', fp);
        writeCsvString(fp, callContext);
        fputc(',', fp);
        writeCsvString(fp, transportKey);
        fprintf(fp, ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
                event->startNs, event->endNs, event->endNs - event->startNs);
    }

    if(fclose(fp) != 0) {
        fprintf(stderr, "Could not close BFS trace %s: %s\n",
                trace->outputPath, strerror(errno));
        free(contexts);
        return false;
    }
    free(contexts);
    return true;
}

void bfsHostTraceDestroy(struct BfsHostTrace* trace) {
    if(trace == NULL) {
        return;
    }
    free(trace->rankOrdinals);
    free(trace->dpuIdsInRank);
    free(trace->sdkPhysicalRankIds);
    free(trace->sdkSliceIds);
    free(trace->sdkMemberIds);
    free(trace->dpuCopyToCallCounts);
    free(trace->dpuCopyFromCallCounts);
    free(trace->events);
    memset(trace, 0, sizeof(*trace));
}
