#define _GNU_SOURCE

#include "host_trace.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static bool parseRepeatId(const char* value, uint64_t* repeatId) {
    char* end = NULL;
    unsigned long long parsed;

    if(value == NULL || value[0] == '\0') {
        *repeatId = 0;
        return true;
    }

    errno = 0;
    parsed = strtoull(value, &end, 10);
    if(errno != 0 || end == value || *end != '\0') {
        fprintf(stderr, "Invalid SPMV_TRACE_REPEAT_ID: %s\n", value);
        return false;
    }
    *repeatId = (uint64_t)parsed;
    return true;
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

bool spmvHostTraceInit(
    struct SpmvHostTrace* trace,
    struct dpu_set_t dpuSet,
    uint32_t configuredDpus,
    uint32_t numTasklets
) {
    const char* outputPath = getenv("SPMV_TRACE_CSV");
    const char* runId = getenv("SPMV_TRACE_RUN_ID");
    const char* repeatId = getenv("SPMV_TRACE_REPEAT_ID");
    struct dpu_set_t rank;
    struct dpu_set_t dpu;
    uint32_t rankOrdinal;
    uint32_t dpuIdInRank;
    uint32_t globalDpuId = 0;

    memset(trace, 0, sizeof(*trace));
    if(outputPath == NULL || outputPath[0] == '\0') {
        return true;
    }

    trace->outputPath = outputPath;
    trace->runId = (runId == NULL || runId[0] == '\0') ? "spmv" : runId;
    trace->configuredDpus = configuredDpus;
    trace->numTasklets = numTasklets;
    trace->capacity = (size_t)configuredDpus * 5u + 1u;
    if(!parseRepeatId(repeatId, &trace->repeatId)) {
        return false;
    }
    if(dpu_get_nr_ranks(dpuSet, &trace->actualRanks) != DPU_OK) {
        fprintf(stderr, "Could not query the allocated rank count\n");
        return false;
    }

    trace->rankOrdinals = calloc(configuredDpus, sizeof(*trace->rankOrdinals));
    trace->dpuIdsInRank = calloc(configuredDpus, sizeof(*trace->dpuIdsInRank));
    trace->events = calloc(trace->capacity, sizeof(*trace->events));
    if(trace->rankOrdinals == NULL || trace->dpuIdsInRank == NULL || trace->events == NULL) {
        fprintf(stderr, "Could not allocate the SpMV host trace buffers\n");
        spmvHostTraceDestroy(trace);
        return false;
    }

    DPU_RANK_FOREACH(dpuSet, rank, rankOrdinal) {
        DPU_FOREACH(rank, dpu, dpuIdInRank) {
            if(globalDpuId >= configuredDpus) {
                fprintf(stderr, "Allocated DPU topology exceeds configured DPU count\n");
                spmvHostTraceDestroy(trace);
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
        spmvHostTraceDestroy(trace);
        return false;
    }

    trace->enabled = true;
    return true;
}

bool spmvHostTraceEnabled(const struct SpmvHostTrace* trace) {
    return trace != NULL && trace->enabled;
}

uint64_t spmvHostTraceNowNs(void) {
    struct timespec now;
    if(clock_gettime(CLOCK_MONOTONIC_RAW, &now) != 0) {
        perror("clock_gettime(CLOCK_MONOTONIC_RAW)");
        exit(EXIT_FAILURE);
    }
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
}

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
) {
    struct SpmvHostTraceEvent* event;

    if(!spmvHostTraceEnabled(trace)) {
        return;
    }
    if(trace->numEvents >= trace->capacity) {
        fprintf(stderr, "SpMV host trace capacity exceeded (%zu events)\n", trace->capacity);
        exit(EXIT_FAILURE);
    }
    if(hasDpu && globalDpuId >= trace->configuredDpus) {
        fprintf(stderr, "SpMV host trace DPU ID %u is out of range\n", globalDpuId);
        exit(EXIT_FAILURE);
    }
    if(endNs < startNs) {
        fprintf(stderr, "SpMV host trace clock moved backwards\n");
        exit(EXIT_FAILURE);
    }

    event = &trace->events[trace->numEvents];
    event->eventId = trace->numEvents;
    event->op = op;
    event->subop = subop;
    event->direction = direction;
    event->hasDpu = hasDpu;
    event->globalDpuId = globalDpuId;
    event->logicalBytes = logicalBytes;
    event->transferBytes = transferBytes;
    event->offsetBytes = offsetBytes;
    event->startNs = startNs;
    event->endNs = endNs;
    ++trace->numEvents;
}

bool spmvHostTraceWrite(const struct SpmvHostTrace* trace) {
    FILE* fp;
    size_t i;

    if(!spmvHostTraceEnabled(trace)) {
        return true;
    }
    fp = fopen(trace->outputPath, "w");
    if(fp == NULL) {
        fprintf(stderr, "Could not open SpMV trace %s: %s\n",
                trace->outputPath, strerror(errno));
        return false;
    }

    fputs("run_id,repeat_id,event_id,configured_dpus,actual_ranks,num_tasklets,"
          "op,subop,direction,global_dpu_id,rank_ordinal,dpu_id_in_rank,"
          "target_space,target_symbol,offset_bytes,logical_bytes,transfer_bytes,"
          "host_start_ns,host_end_ns,measured_ns\n", fp);

    for(i = 0; i < trace->numEvents; ++i) {
        const struct SpmvHostTraceEvent* event = &trace->events[i];

        writeCsvString(fp, trace->runId);
        fprintf(fp, ",%" PRIu64 ",%" PRIu64 ",%u,%u,%u,",
                trace->repeatId, event->eventId, trace->configuredDpus,
                trace->actualRanks, trace->numTasklets);
        writeCsvString(fp, event->op);
        fputc(',', fp);
        writeCsvString(fp, event->subop);
        fputc(',', fp);
        writeCsvString(fp, event->direction);
        fputc(',', fp);
        if(event->hasDpu) {
            fprintf(fp, "%u,%u,%u", event->globalDpuId,
                    trace->rankOrdinals[event->globalDpuId],
                    trace->dpuIdsInRank[event->globalDpuId]);
        } else {
            fputs(",,", fp);
        }
        if(event->hasDpu) {
            fputs(",MRAM,DPU_MRAM_HEAP_POINTER_NAME,", fp);
            fprintf(fp, "%" PRIu64 ",%" PRIu64 ",%" PRIu64,
                    event->offsetBytes, event->logicalBytes, event->transferBytes);
        } else {
            fputs(",,,,,", fp);
        }
        fprintf(fp, ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
                event->startNs, event->endNs, event->endNs - event->startNs);
    }

    if(fclose(fp) != 0) {
        fprintf(stderr, "Could not close SpMV trace %s: %s\n",
                trace->outputPath, strerror(errno));
        return false;
    }
    return true;
}

void spmvHostTraceDestroy(struct SpmvHostTrace* trace) {
    if(trace == NULL) {
        return;
    }
    free(trace->rankOrdinals);
    free(trace->dpuIdsInRank);
    free(trace->events);
    memset(trace, 0, sizeof(*trace));
}
