#define _GNU_SOURCE

#include "context_probe.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../support/utils.h"

enum ProbeCondition {
    COPY_TO_THEN_COPY_TO = 0,
    COPY_FROM_THEN_COPY_TO = 1,
    LAUNCH_COPY_FROM_THEN_COPY_TO = 2,
    NUM_PROBE_CONDITIONS = 3,
};

struct ProbeConfig {
    const char* outputPath;
    const char* runId;
    uint64_t processRepeat;
    uint64_t samples;
    uint64_t warmups;
    uint32_t targetGlobalDpuId;
};

static volatile uint64_t sourceTouchSink;

static uint64_t nowNs(void) {
    struct timespec now;
    if(clock_gettime(CLOCK_MONOTONIC_RAW, &now) != 0) {
        perror("clock_gettime(CLOCK_MONOTONIC_RAW)");
        exit(EXIT_FAILURE);
    }
    return (uint64_t)now.tv_sec * UINT64_C(1000000000)
        + (uint64_t)now.tv_nsec;
}

static bool parseUnsigned(
    const char* name,
    const char* value,
    uint64_t defaultValue,
    uint64_t* parsedValue
) {
    char* end = NULL;
    unsigned long long parsed;

    if(value == NULL || value[0] == '\0') {
        *parsedValue = defaultValue;
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

static bool loadConfig(struct ProbeConfig* config) {
    uint64_t targetDpu;

    memset(config, 0, sizeof(*config));
    config->outputPath = getenv("BFS_CONTEXT_PROBE_CSV");
    config->runId = getenv("BFS_CONTEXT_PROBE_RUN_ID");
    if(config->outputPath == NULL || config->outputPath[0] == '\0') {
        return false;
    }
    if(config->runId == NULL || config->runId[0] == '\0') {
        config->runId = "bfs_context_probe";
    }
    if(!parseUnsigned("BFS_CONTEXT_PROBE_PROCESS_REPEAT",
                      getenv("BFS_CONTEXT_PROBE_PROCESS_REPEAT"), 0,
                      &config->processRepeat)
       || !parseUnsigned("BFS_CONTEXT_PROBE_SAMPLES",
                         getenv("BFS_CONTEXT_PROBE_SAMPLES"), 60,
                         &config->samples)
       || !parseUnsigned("BFS_CONTEXT_PROBE_WARMUPS",
                         getenv("BFS_CONTEXT_PROBE_WARMUPS"), 5,
                         &config->warmups)
       || !parseUnsigned("BFS_CONTEXT_PROBE_DPU",
                         getenv("BFS_CONTEXT_PROBE_DPU"), 0, &targetDpu)) {
        return false;
    }
    if(config->samples == 0 || targetDpu > UINT32_MAX) {
        fprintf(stderr, "Context probe requires samples > 0 and a uint32 DPU ID\n");
        return false;
    }
    config->targetGlobalDpuId = (uint32_t)targetDpu;
    return true;
}

bool bfsContextProbeRequested(void) {
    const char* outputPath = getenv("BFS_CONTEXT_PROBE_CSV");
    return outputPath != NULL && outputPath[0] != '\0';
}

static const char* conditionName(enum ProbeCondition condition) {
    switch(condition) {
        case COPY_TO_THEN_COPY_TO:
            return "COPY_TO_THEN_COPY_TO";
        case COPY_FROM_THEN_COPY_TO:
            return "COPY_FROM_THEN_COPY_TO";
        case LAUNCH_COPY_FROM_THEN_COPY_TO:
            return "LAUNCH_COPY_FROM_THEN_COPY_TO";
        default:
            return "UNKNOWN";
    }
}

static const char* predecessorChain(enum ProbeCondition condition) {
    switch(condition) {
        case COPY_TO_THEN_COPY_TO:
            return "COPY_TO>MEASURED_COPY_TO";
        case COPY_FROM_THEN_COPY_TO:
            return "COPY_TO_SETUP>COPY_FROM>MEASURED_COPY_TO";
        case LAUNCH_COPY_FROM_THEN_COPY_TO:
            return "COPY_TO_SETUP>LAUNCH>COPY_FROM>MEASURED_COPY_TO";
        default:
            return "UNKNOWN";
    }
}

static bool findTargetDpu(
    struct dpu_set_t dpuSet,
    uint32_t targetGlobalDpuId,
    struct dpu_set_t* targetDpu,
    uint32_t* targetRankOrdinal,
    uint32_t* targetDpuIdInRank
) {
    struct dpu_set_t rank;
    struct dpu_set_t dpu;
    uint32_t rankOrdinal;
    uint32_t dpuIdInRank;
    uint32_t globalDpuId = 0;

    DPU_RANK_FOREACH(dpuSet, rank, rankOrdinal) {
        DPU_FOREACH(rank, dpu, dpuIdInRank) {
            if(globalDpuId == targetGlobalDpuId) {
                *targetDpu = dpu;
                *targetRankOrdinal = rankOrdinal;
                *targetDpuIdInRank = dpuIdInRank;
                return true;
            }
            ++globalDpuId;
        }
    }
    return false;
}

static void copyTo(
    struct dpu_set_t dpu,
    uint32_t offset,
    const uint8_t* source,
    uint32_t size
) {
    DPU_ASSERT(dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME, offset,
                           source, size));
}

static void copyFrom(
    struct dpu_set_t dpu,
    uint32_t offset,
    uint8_t* destination,
    uint32_t size
) {
    DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME, offset,
                             destination, size));
}

static void normalizeAllFrontiers(
    struct dpu_set_t dpuSet,
    const struct DPUParams* dpuParams,
    const uint8_t* zeroBuffer,
    uint32_t transferBytes
) {
    struct dpu_set_t dpu;
    uint32_t dpuIdx;

    DPU_FOREACH(dpuSet, dpu, dpuIdx) {
        copyTo(dpu, dpuParams[dpuIdx].dpuNextFrontier_m,
               zeroBuffer, transferBytes);
    }
}

static void touchSourceBuffer(const uint8_t* sourceBuffer, uint32_t size) {
    uint64_t checksum = 0;
    uint32_t offset;

    for(offset = 0; offset < size; offset += 64) {
        checksum += sourceBuffer[offset];
    }
    checksum += sourceBuffer[size - 1];
    sourceTouchSink ^= checksum;
}

static bool runOneCondition(
    FILE* output,
    const struct ProbeConfig* config,
    struct dpu_set_t dpuSet,
    struct dpu_set_t targetDpu,
    uint32_t configuredDpus,
    uint32_t actualRanks,
    uint32_t targetRankOrdinal,
    uint32_t targetDpuIdInRank,
    uint32_t targetOffset,
    uint32_t transferBytes,
    const uint8_t* zeroBuffer,
    const uint8_t* sourceBuffer,
    uint8_t* readbackBuffer,
    uint64_t sampleIndex,
    uint32_t orderIndex,
    enum ProbeCondition condition,
    bool record
) {
    uint64_t launchStartNs = 0;
    uint64_t launchEndNs = 0;
    uint64_t predecessorStartNs;
    uint64_t predecessorEndNs;
    uint64_t sourcePretouchStartNs;
    uint64_t sourcePretouchEndNs;
    uint64_t measuredStartNs;
    uint64_t measuredEndNs;
    bool verified;
    const char* previousOp;
    const char* previousDirection;
    unsigned int afterLaunch;
    unsigned int directionSwitched;

    if(condition == COPY_TO_THEN_COPY_TO) {
        predecessorStartNs = nowNs();
        copyTo(targetDpu, targetOffset, zeroBuffer, transferBytes);
        predecessorEndNs = nowNs();
        previousOp = "dpu_copy_to";
        previousDirection = "TO_DPU";
        afterLaunch = 0;
        directionSwitched = 0;
    } else if(condition == COPY_FROM_THEN_COPY_TO) {
        copyTo(targetDpu, targetOffset, zeroBuffer, transferBytes);
        predecessorStartNs = nowNs();
        copyFrom(targetDpu, targetOffset, readbackBuffer, transferBytes);
        predecessorEndNs = nowNs();
        previousOp = "dpu_copy_from";
        previousDirection = "FROM_DPU";
        afterLaunch = 0;
        directionSwitched = 1;
    } else {
        copyTo(targetDpu, targetOffset, zeroBuffer, transferBytes);
        launchStartNs = nowNs();
        DPU_ASSERT(dpu_launch(dpuSet, DPU_SYNCHRONOUS));
        launchEndNs = nowNs();
        predecessorStartNs = nowNs();
        copyFrom(targetDpu, targetOffset, readbackBuffer, transferBytes);
        predecessorEndNs = nowNs();
        previousOp = "dpu_copy_from";
        previousDirection = "FROM_DPU";
        afterLaunch = 1;
        directionSwitched = 1;
    }

    sourcePretouchStartNs = nowNs();
    touchSourceBuffer(sourceBuffer, transferBytes);
    sourcePretouchEndNs = nowNs();
    measuredStartNs = nowNs();
    copyTo(targetDpu, targetOffset, sourceBuffer, transferBytes);
    measuredEndNs = nowNs();

    memset(readbackBuffer, 0, transferBytes);
    copyFrom(targetDpu, targetOffset, readbackBuffer, transferBytes);
    verified = memcmp(sourceBuffer, readbackBuffer, transferBytes) == 0;

    if(record) {
        fprintf(
            output,
            "%s,%" PRIu64 ",%u,%u,%u,%u,%u,%u,%" PRIu64 ",%u,%s,%s,"
            "dpu_copy_to,TO_DPU,SINGLE_COPY,MRAM,%u,%u,SHARED_FIXED_BUFFER,1,"
            "%s,%s,%u,%u,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
            ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%s\n",
            config->runId, config->processRepeat, configuredDpus, NR_TASKLETS,
            actualRanks, config->targetGlobalDpuId, targetRankOrdinal,
            targetDpuIdInRank, sampleIndex, orderIndex, conditionName(condition),
            predecessorChain(condition), transferBytes, targetOffset,
            previousOp, previousDirection, directionSwitched, afterLaunch,
            predecessorEndNs - predecessorStartNs,
            launchEndNs - launchStartNs,
            measuredStartNs - predecessorEndNs,
            sourcePretouchEndNs - sourcePretouchStartNs,
            measuredStartNs, measuredEndNs,
            measuredEndNs - measuredStartNs, verified ? "ok" : "fail"
        );
    }
    if(!verified) {
        fprintf(stderr,
                "Context probe readback mismatch for sample %" PRIu64
                ", condition %s\n",
                sampleIndex, conditionName(condition));
    }
    return verified;
}

bool bfsRunContextProbe(
    struct dpu_set_t dpuSet,
    uint32_t configuredDpus,
    const struct DPUParams* dpuParams,
    uint32_t numNodes
) {
    struct ProbeConfig config;
    struct dpu_set_t targetDpu;
    uint32_t targetRankOrdinal;
    uint32_t targetDpuIdInRank;
    uint32_t actualRanks;
    uint32_t transferBytes;
    uint32_t targetOffset;
    uint8_t* zeroBuffer;
    uint8_t* sourceBuffer;
    uint8_t* readbackBuffer;
    FILE* output;
    uint64_t cycle;
    uint32_t orderIndex;
    bool success = true;

    if(!loadConfig(&config)) {
        return false;
    }
    if(config.targetGlobalDpuId >= configuredDpus
       || dpuParams[config.targetGlobalDpuId].dpuNumNodes == 0) {
        fprintf(stderr, "Context probe target DPU %u is inactive or out of range\n",
                config.targetGlobalDpuId);
        return false;
    }
    if(numNodes == 0 || numNodes % 64 != 0) {
        fprintf(stderr, "Context probe requires a positive node count divisible by 64\n");
        return false;
    }
    if(!findTargetDpu(dpuSet, config.targetGlobalDpuId, &targetDpu,
                      &targetRankOrdinal, &targetDpuIdInRank)) {
        fprintf(stderr, "Could not locate context probe DPU %u\n",
                config.targetGlobalDpuId);
        return false;
    }
    DPU_ASSERT(dpu_get_nr_ranks(dpuSet, &actualRanks));

    transferBytes = numNodes / 64 * sizeof(uint64_t);
    targetOffset = dpuParams[config.targetGlobalDpuId].dpuNextFrontier_m;
    zeroBuffer = calloc(transferBytes, 1);
    sourceBuffer = malloc(transferBytes);
    readbackBuffer = malloc(transferBytes);
    if(zeroBuffer == NULL || sourceBuffer == NULL || readbackBuffer == NULL) {
        fprintf(stderr, "Could not allocate context probe host buffers\n");
        free(zeroBuffer);
        free(sourceBuffer);
        free(readbackBuffer);
        return false;
    }
    for(uint32_t i = 0; i < transferBytes; ++i) {
        sourceBuffer[i] = (uint8_t)(UINT8_C(0xa5) ^ (uint8_t)(i & 0x3f));
    }

    output = fopen(config.outputPath, "w");
    if(output == NULL) {
        fprintf(stderr, "Could not open context probe CSV %s: %s\n",
                config.outputPath, strerror(errno));
        free(zeroBuffer);
        free(sourceBuffer);
        free(readbackBuffer);
        return false;
    }
    fputs(
        "run_id,process_repeat,configured_dpus,num_tasklets,actual_ranks,"
        "target_global_dpu_id,rank_ordinal,dpu_id_in_rank,sample_index,"
        "order_index,condition,predecessor_chain,op,direction,sdk_api_kind,"
        "target_space,transfer_bytes_per_dpu,offset_bytes,source_buffer_class,"
        "same_source_across_conditions,previous_op,previous_direction,"
        "direction_switched,after_launch,predecessor_ns,launch_ns,"
        "ns_since_previous_sdk_event,source_pretouch_ns,host_start_ns,"
        "host_end_ns,measured_ns,verification\n",
        output
    );

    normalizeAllFrontiers(dpuSet, dpuParams, zeroBuffer, transferBytes);

    for(cycle = 0; cycle < config.warmups && success; ++cycle) {
        for(orderIndex = 0; orderIndex < NUM_PROBE_CONDITIONS; ++orderIndex) {
            enum ProbeCondition condition =
                (enum ProbeCondition)((cycle + orderIndex) % NUM_PROBE_CONDITIONS);
            success = runOneCondition(
                output, &config, dpuSet, targetDpu, configuredDpus, actualRanks,
                targetRankOrdinal, targetDpuIdInRank, targetOffset, transferBytes,
                zeroBuffer, sourceBuffer, readbackBuffer, cycle, orderIndex,
                condition, false
            );
            if(!success) {
                break;
            }
        }
    }
    for(cycle = 0; cycle < config.samples && success; ++cycle) {
        for(orderIndex = 0; orderIndex < NUM_PROBE_CONDITIONS; ++orderIndex) {
            enum ProbeCondition condition =
                (enum ProbeCondition)((cycle + orderIndex) % NUM_PROBE_CONDITIONS);
            success = runOneCondition(
                output, &config, dpuSet, targetDpu, configuredDpus, actualRanks,
                targetRankOrdinal, targetDpuIdInRank, targetOffset, transferBytes,
                zeroBuffer, sourceBuffer, readbackBuffer, cycle, orderIndex,
                condition, true
            );
            if(!success) {
                break;
            }
        }
    }

    if(fclose(output) != 0) {
        fprintf(stderr, "Could not close context probe CSV %s\n", config.outputPath);
        success = false;
    }
    free(zeroBuffer);
    free(sourceBuffer);
    free(readbackBuffer);

    if(success) {
        printf("Context probe wrote %" PRIu64 " samples per condition to %s\n",
               config.samples, config.outputPath);
    }
    return success;
}
