#define _GNU_SOURCE

#include "host_trace.h"

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
    const char* eventSdkApiKind,
    const char* distributionClass,
    const char* sameSource,
    const char* eventPhaseClass,
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
        "v2;op=%s;direction=%s;sdk_api_kind=%s;"
        "logical_distribution_class=%s;target_space=MRAM;"
        "transfer_bytes_per_dpu=%" PRIu64
        ";active_dpus=1;active_ranks=1;active_dpus_per_rank=1;"
        "rank_ordinal=%u;dpu_id_in_rank=%u;same_source_across_group=%s;"
        "phase_class=%s",
        event->op, event->direction, eventSdkApiKind, distributionClass,
        event->transferBytes, trace->rankOrdinals[event->globalDpuId],
        trace->dpuIdsInRank[event->globalDpuId], sameSource, eventPhaseClass
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
    struct dpu_set_t rank;
    struct dpu_set_t dpu;
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
    trace->dpuCopyToCallCounts = calloc(configuredDpus,
                                        sizeof(*trace->dpuCopyToCallCounts));
    trace->dpuCopyFromCallCounts = calloc(configuredDpus,
                                          sizeof(*trace->dpuCopyFromCallCounts));
    trace->events = calloc(trace->capacity, sizeof(*trace->events));
    if(trace->rankOrdinals == NULL || trace->dpuIdsInRank == NULL
       || trace->dpuCopyToCallCounts == NULL
       || trace->dpuCopyFromCallCounts == NULL || trace->events == NULL) {
        fprintf(stderr, "Could not allocate the BFS host trace buffers\n");
        bfsHostTraceDestroy(trace);
        return false;
    }

    DPU_RANK_FOREACH(dpuSet, rank, rankOrdinal) {
        DPU_FOREACH(rank, dpu, dpuIdInRank) {
            if(globalDpuId >= configuredDpus) {
                fprintf(stderr, "Allocated DPU topology exceeds configured DPU count\n");
                bfsHostTraceDestroy(trace);
                return false;
            }
            trace->rankOrdinals[globalDpuId] = rankOrdinal;
            trace->dpuIdsInRank[globalDpuId] = dpuIdInRank;
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

    fputs("run_id,repeat_id,event_id,configured_dpus,actual_ranks,num_tasklets,"
          "op,direction,sdk_api_kind,logical_distribution_class,target_space,"
          "transfer_bytes_per_dpu,active_dpus,active_ranks,active_dpus_per_rank,"
          "rank_ordinal,dpu_id_in_rank,same_source_across_group,phase_class,"
          "subop,bfs_level,global_dpu_id,target_symbol,offset_bytes,offset_feature,"
          "logical_bytes,transfer_bytes,"
          "op_call_index,dpu_op_call_index,process_state,pretrace_warmup_runs,"
          "host_numa_node,call_context,transport_key,"
          "host_start_ns,host_end_ns,measured_ns\n", fp);

    for(i = 0; i < trace->numEvents; ++i) {
        const struct BfsHostTraceEvent* event = &trace->events[i];
        const char* eventSdkApiKind = sdkApiKind(event);
        const char* distributionClass = logicalDistributionClass(event);
        const char* sameSource = sameSourceAcrossGroup(event);
        const char* eventPhaseClass = phaseClass(event);
        char offsetFeature[160];
        char callContext[256];
        char transportKey[1024];

        formatOffsetFeature(event, offsetFeature, sizeof(offsetFeature));
        formatCallContext(trace, event, callContext, sizeof(callContext));
        if(!formatTransportKey(trace, event, eventSdkApiKind,
                              distributionClass, sameSource, eventPhaseClass,
                              transportKey,
                              sizeof(transportKey))) {
            fprintf(stderr, "BFS transport key is too long for event %" PRIu64 "\n",
                    event->eventId);
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
            fprintf(fp, "MRAM,%" PRIu64 ",1,1,1,%u,%u,",
                    event->transferBytes,
                    trace->rankOrdinals[event->globalDpuId],
                    trace->dpuIdsInRank[event->globalDpuId]);
            writeCsvString(fp, sameSource);
        } else {
            fputs(",,,,,,,", fp);
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
            fprintf(fp, ",%" PRIu64 ",%" PRIu64,
                    event->logicalBytes, event->transferBytes);
        } else {
            fputs(",,,none,,", fp);
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
        return false;
    }
    return true;
}

void bfsHostTraceDestroy(struct BfsHostTrace* trace) {
    if(trace == NULL) {
        return;
    }
    free(trace->rankOrdinals);
    free(trace->dpuIdsInRank);
    free(trace->dpuCopyToCallCounts);
    free(trace->dpuCopyFromCallCounts);
    free(trace->events);
    memset(trace, 0, sizeof(*trace));
}
