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
        fprintf(stderr, "Invalid RED_TRACE_REPEAT_ID: %s\n", value);
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

static void growEvents(struct RedHostTrace* trace) {
    size_t newCapacity;
    struct RedHostTraceEvent* newEvents;

    if(trace->eventCapacity > SIZE_MAX / (2u * sizeof(*trace->events))) {
        fprintf(stderr, "RED host event trace capacity overflow\n");
        exit(EXIT_FAILURE);
    }
    newCapacity = trace->eventCapacity == 0 ? 64u : trace->eventCapacity * 2u;
    newEvents = realloc(trace->events, newCapacity * sizeof(*trace->events));
    if(newEvents == NULL) {
        fprintf(stderr, "Could not grow the RED event trace to %zu rows\n",
                newCapacity);
        exit(EXIT_FAILURE);
    }
    memset(newEvents + trace->eventCapacity, 0,
           (newCapacity - trace->eventCapacity) * sizeof(*newEvents));
    trace->events = newEvents;
    trace->eventCapacity = newCapacity;
}

static void growDpuRows(struct RedHostTrace* trace) {
    size_t newCapacity;
    struct RedHostTraceDpu* newRows;

    if(trace->dpuRowCapacity > SIZE_MAX / (2u * sizeof(*trace->dpuRows))) {
        fprintf(stderr, "RED host DPU trace capacity overflow\n");
        exit(EXIT_FAILURE);
    }
    newCapacity = trace->dpuRowCapacity == 0
        ? (size_t)trace->configuredDpus
        : trace->dpuRowCapacity * 2u;
    newRows = realloc(trace->dpuRows, newCapacity * sizeof(*trace->dpuRows));
    if(newRows == NULL) {
        fprintf(stderr, "Could not grow the RED DPU trace to %zu rows\n",
                newCapacity);
        exit(EXIT_FAILURE);
    }
    memset(newRows + trace->dpuRowCapacity, 0,
           (newCapacity - trace->dpuRowCapacity) * sizeof(*newRows));
    trace->dpuRows = newRows;
    trace->dpuRowCapacity = newCapacity;
}

bool redHostTraceRequested(void) {
    const char* outputPath = getenv("RED_TRACE_CSV");
    return outputPath != NULL && outputPath[0] != '\0';
}

bool redHostTraceInit(
    struct RedHostTrace* trace,
    struct dpu_set_t dpuSet,
    uint32_t configuredDpus,
    uint32_t numTasklets,
    uint64_t totalInputElements,
    uint64_t totalInputBytes
) {
    const char* eventsOutputPath = getenv("RED_TRACE_CSV");
    const char* dpusOutputPath = getenv("RED_TRACE_DPUS_CSV");
    const char* runId = getenv("RED_TRACE_RUN_ID");
    const char* repeatId = getenv("RED_TRACE_REPEAT_ID");
    struct dpu_set_t rank;
    struct dpu_set_t dpu;
    uint32_t rankOrdinal;
    uint32_t dpuIdInRank;
    uint32_t globalDpuId = 0;

    memset(trace, 0, sizeof(*trace));
    if(!redHostTraceRequested()) {
        return true;
    }
    if(dpusOutputPath == NULL || dpusOutputPath[0] == '\0') {
        fprintf(stderr, "RED_TRACE_DPUS_CSV is required with RED_TRACE_CSV\n");
        return false;
    }

    trace->eventsOutputPath = eventsOutputPath;
    trace->dpusOutputPath = dpusOutputPath;
    trace->runId = (runId == NULL || runId[0] == '\0') ? "red" : runId;
    trace->configuredDpus = configuredDpus;
    trace->numTasklets = numTasklets;
    trace->totalInputElements = totalInputElements;
    trace->totalInputBytes = totalInputBytes;
    trace->eventCapacity = 32u;
    trace->dpuRowCapacity = (size_t)configuredDpus * 12u;
    if(!parseRepeatId(repeatId, &trace->repeatId)) {
        return false;
    }
    if(dpu_get_nr_ranks(dpuSet, &trace->actualRanks) != DPU_OK) {
        fprintf(stderr, "Could not query the allocated rank count\n");
        return false;
    }

    trace->rankOrdinals = calloc(configuredDpus, sizeof(*trace->rankOrdinals));
    trace->dpuIdsInRank = calloc(configuredDpus, sizeof(*trace->dpuIdsInRank));
    trace->events = calloc(trace->eventCapacity, sizeof(*trace->events));
    trace->dpuRows = calloc(trace->dpuRowCapacity, sizeof(*trace->dpuRows));
    if(trace->rankOrdinals == NULL || trace->dpuIdsInRank == NULL
       || trace->events == NULL || trace->dpuRows == NULL) {
        fprintf(stderr, "Could not allocate the RED host trace buffers\n");
        redHostTraceDestroy(trace);
        return false;
    }

    DPU_RANK_FOREACH(dpuSet, rank, rankOrdinal) {
        DPU_FOREACH(rank, dpu, dpuIdInRank) {
            if(globalDpuId >= configuredDpus) {
                fprintf(stderr, "Allocated DPU topology exceeds configured DPU count\n");
                redHostTraceDestroy(trace);
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
        redHostTraceDestroy(trace);
        return false;
    }

    trace->enabled = true;
    return true;
}

bool redHostTraceEnabled(const struct RedHostTrace* trace) {
    return trace != NULL && trace->enabled;
}

uint64_t redHostTraceNowNs(void) {
    struct timespec now;
    if(clock_gettime(CLOCK_MONOTONIC_RAW, &now) != 0) {
        perror("clock_gettime(CLOCK_MONOTONIC_RAW)");
        exit(EXIT_FAILURE);
    }
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
}

void redHostTraceRecordEvent(
    struct RedHostTrace* trace,
    const char* op,
    const char* subop,
    int32_t iteration,
    int32_t warmup,
    uint32_t activeDpus,
    uint64_t startNs,
    uint64_t endNs
) {
    struct RedHostTraceEvent* event;

    if(!redHostTraceEnabled(trace)) {
        return;
    }
    if(trace->numEvents >= trace->eventCapacity) {
        growEvents(trace);
    }
    if(activeDpus > trace->configuredDpus) {
        fprintf(stderr, "RED host trace active DPU count is out of range\n");
        exit(EXIT_FAILURE);
    }
    if(endNs < startNs) {
        fprintf(stderr, "RED host trace clock moved backwards\n");
        exit(EXIT_FAILURE);
    }

    event = &trace->events[trace->numEvents];
    event->eventId = trace->numEvents;
    event->op = op;
    event->subop = subop;
    event->iteration = iteration;
    event->warmup = warmup;
    event->activeDpus = activeDpus;
    event->startNs = startNs;
    event->endNs = endNs;
    ++trace->numEvents;
}

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
) {
    struct RedHostTraceEvent* event;
    uint64_t eventId;
    uint64_t totalLogicalBytes = 0;
    uint64_t totalTransferBytes = 0;
    uint32_t dpuId;

    if(!redHostTraceEnabled(trace)) {
        return;
    }
    if(endNs < startNs) {
        fprintf(stderr, "RED transfer trace clock moved backwards\n");
        exit(EXIT_FAILURE);
    }
    if(sizePerDpuBytes > UINT64_MAX / trace->configuredDpus) {
        fprintf(stderr, "RED transfer byte total overflow\n");
        exit(EXIT_FAILURE);
    }
    for(dpuId = 0; dpuId < trace->configuredDpus; ++dpuId) {
        uint64_t logicalBytes = logicalBytesPerDpu == NULL
            ? uniformLogicalBytes
            : logicalBytesPerDpu[dpuId];
        if(logicalBytes > sizePerDpuBytes) {
            fprintf(stderr, "RED DPU %u logical bytes exceed transfer bytes\n",
                    dpuId);
            exit(EXIT_FAILURE);
        }
        if(totalLogicalBytes > UINT64_MAX - logicalBytes) {
            fprintf(stderr, "RED logical byte total overflow\n");
            exit(EXIT_FAILURE);
        }
        totalLogicalBytes += logicalBytes;
    }
    totalTransferBytes = sizePerDpuBytes * trace->configuredDpus;

    if(trace->numEvents >= trace->eventCapacity) {
        growEvents(trace);
    }
    eventId = trace->numEvents;
    event = &trace->events[eventId];
    event->eventId = eventId;
    event->op = "dpu_transfer";
    event->subop = subop;
    event->direction = direction;
    event->iteration = iteration;
    event->warmup = warmup;
    event->activeDpus = trace->configuredDpus;
    event->hasTransfer = true;
    event->sizePerDpuBytes = sizePerDpuBytes;
    event->totalLogicalBytes = totalLogicalBytes;
    event->totalTransferBytes = totalTransferBytes;
    event->targetSpace = targetSpace;
    event->targetSymbol = targetSymbol;
    event->offsetBytes = offsetBytes;
    event->startNs = startNs;
    event->endNs = endNs;
    ++trace->numEvents;

    for(dpuId = 0; dpuId < trace->configuredDpus; ++dpuId) {
        struct RedHostTraceDpu* row;
        uint64_t logicalBytes = logicalBytesPerDpu == NULL
            ? uniformLogicalBytes
            : logicalBytesPerDpu[dpuId];
        if(trace->numDpuRows >= trace->dpuRowCapacity) {
            growDpuRows(trace);
        }
        row = &trace->dpuRows[trace->numDpuRows];
        row->eventId = eventId;
        row->iteration = iteration;
        row->warmup = warmup;
        row->op = event->op;
        row->subop = subop;
        row->direction = direction;
        row->globalDpuId = dpuId;
        row->logicalBytes = logicalBytes;
        row->transferBytes = sizePerDpuBytes;
        ++trace->numDpuRows;
    }
}

static bool writeEvents(const struct RedHostTrace* trace) {
    FILE* fp;
    size_t i;

    fp = fopen(trace->eventsOutputPath, "w");
    if(fp == NULL) {
        fprintf(stderr, "Could not open RED event trace %s: %s\n",
                trace->eventsOutputPath, strerror(errno));
        return false;
    }
    fputs("run_id,repeat_id,event_id,configured_dpus,actual_ranks,num_tasklets,"
          "total_input_elements,total_input_bytes,iteration,warmup,"
          "op,subop,direction,active_dpus,size_per_dpu_bytes,"
          "total_logical_bytes,total_transfer_bytes,target_space,target_symbol,"
          "offset_bytes,host_start_ns,host_end_ns,measured_ns\n", fp);

    for(i = 0; i < trace->numEvents; ++i) {
        const struct RedHostTraceEvent* event = &trace->events[i];

        writeCsvString(fp, trace->runId);
        fprintf(fp, ",%" PRIu64 ",%" PRIu64 ",%u,%u,%u,%" PRIu64
                ",%" PRIu64 ",",
                trace->repeatId, event->eventId, trace->configuredDpus,
                trace->actualRanks, trace->numTasklets,
                trace->totalInputElements, trace->totalInputBytes);
        if(event->iteration >= 0) {
            fprintf(fp, "%" PRId32, event->iteration);
        }
        fputc(',', fp);
        if(event->warmup >= 0) {
            fprintf(fp, "%" PRId32, event->warmup);
        }
        fputc(',', fp);
        writeCsvString(fp, event->op);
        fputc(',', fp);
        writeCsvString(fp, event->subop);
        fputc(',', fp);
        writeCsvString(fp, event->direction);
        fprintf(fp, ",%u,", event->activeDpus);
        if(event->hasTransfer) {
            fprintf(fp, "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",",
                    event->sizePerDpuBytes, event->totalLogicalBytes,
                    event->totalTransferBytes);
            writeCsvString(fp, event->targetSpace);
            fputc(',', fp);
            writeCsvString(fp, event->targetSymbol);
            fprintf(fp, ",%" PRIu64, event->offsetBytes);
        } else {
            fputs(",,,,,", fp);
        }
        fprintf(fp, ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
                event->startNs, event->endNs, event->endNs - event->startNs);
    }

    if(fclose(fp) != 0) {
        fprintf(stderr, "Could not close RED event trace %s: %s\n",
                trace->eventsOutputPath, strerror(errno));
        return false;
    }
    return true;
}

static bool writeDpuRows(const struct RedHostTrace* trace) {
    FILE* fp;
    size_t i;

    fp = fopen(trace->dpusOutputPath, "w");
    if(fp == NULL) {
        fprintf(stderr, "Could not open RED DPU trace %s: %s\n",
                trace->dpusOutputPath, strerror(errno));
        return false;
    }
    fputs("run_id,repeat_id,event_id,configured_dpus,actual_ranks,num_tasklets,"
          "iteration,warmup,op,subop,direction,global_dpu_id,rank_ordinal,"
          "dpu_id_in_rank,logical_bytes,transfer_bytes,kernel_cycles\n", fp);

    for(i = 0; i < trace->numDpuRows; ++i) {
        const struct RedHostTraceDpu* row = &trace->dpuRows[i];

        writeCsvString(fp, trace->runId);
        fprintf(fp, ",%" PRIu64 ",%" PRIu64 ",%u,%u,%u,%" PRId32
                ",%" PRId32 ",",
                trace->repeatId, row->eventId, trace->configuredDpus,
                trace->actualRanks, trace->numTasklets,
                row->iteration, row->warmup);
        writeCsvString(fp, row->op);
        fputc(',', fp);
        writeCsvString(fp, row->subop);
        fputc(',', fp);
        writeCsvString(fp, row->direction);
        fprintf(fp, ",%u,%u,%u,%" PRIu64 ",%" PRIu64 ",\n",
                row->globalDpuId,
                trace->rankOrdinals[row->globalDpuId],
                trace->dpuIdsInRank[row->globalDpuId],
                row->logicalBytes, row->transferBytes);
    }

    if(fclose(fp) != 0) {
        fprintf(stderr, "Could not close RED DPU trace %s: %s\n",
                trace->dpusOutputPath, strerror(errno));
        return false;
    }
    return true;
}

bool redHostTraceWrite(const struct RedHostTrace* trace) {
    if(!redHostTraceEnabled(trace)) {
        return true;
    }
    if(!writeEvents(trace)) {
        return false;
    }
    return writeDpuRows(trace);
}

void redHostTraceDestroy(struct RedHostTrace* trace) {
    if(trace == NULL) {
        return;
    }
    free(trace->rankOrdinals);
    free(trace->dpuIdsInRank);
    free(trace->events);
    free(trace->dpuRows);
    memset(trace, 0, sizeof(*trace));
}
