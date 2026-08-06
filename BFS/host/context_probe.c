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

enum GroupProbeCondition {
    H2D_GROUP_THEN_H2D_GROUP = 0,
    D2H_GROUP_THEN_H2D_GROUP = 1,
    LAUNCH_D2H_GROUP_THEN_H2D_GROUP = 2,
    NUM_GROUP_PROBE_CONDITIONS = 3,
};

struct GroupProbeConfig {
    const char* outputPath;
    const char* runId;
    uint64_t processRepeat;
    uint64_t samples;
    uint64_t warmups;
};

struct GroupProbeDpu {
    struct dpu_set_t dpu;
    uint32_t globalDpuId;
    uint32_t rankOrdinal;
    uint32_t dpuIdInRank;
    uint32_t targetOffset;
};

static volatile uint64_t sourceTouchSink;

#define PROBE_HOST_BUFFER_ALIGNMENT UINT32_C(4096)

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

static uint64_t hashBuffer(const uint8_t* buffer, uint32_t size) {
    uint64_t hash = UINT64_C(14695981039346656037);

    for(uint32_t i = 0; i < size; ++i) {
        hash ^= buffer[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint8_t* allocateAlignedBuffer(uint32_t size) {
    void* buffer = NULL;
    int error = posix_memalign(&buffer, PROBE_HOST_BUFFER_ALIGNMENT, size);

    if(error != 0) {
        errno = error;
        return NULL;
    }
    return (uint8_t*)buffer;
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
    uint64_t sourceContentHash,
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
            "0x%" PRIxPTR ",0x%016" PRIx64 ",%u,ZERO_WRITTEN,"
            "%s,%s,%u,%u,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
            ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%s\n",
            config->runId, config->processRepeat, configuredDpus, NR_TASKLETS,
            actualRanks, config->targetGlobalDpuId, targetRankOrdinal,
            targetDpuIdInRank, sampleIndex, orderIndex, conditionName(condition),
            predecessorChain(condition), transferBytes, targetOffset,
            (uintptr_t)sourceBuffer, sourceContentHash,
            PROBE_HOST_BUFFER_ALIGNMENT,
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
    uint64_t sourceContentHash;
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
    zeroBuffer = allocateAlignedBuffer(transferBytes);
    sourceBuffer = allocateAlignedBuffer(transferBytes);
    readbackBuffer = allocateAlignedBuffer(transferBytes);
    if(zeroBuffer == NULL || sourceBuffer == NULL || readbackBuffer == NULL) {
        fprintf(stderr, "Could not allocate context probe host buffers\n");
        free(zeroBuffer);
        free(sourceBuffer);
        free(readbackBuffer);
        return false;
    }
    memset(zeroBuffer, 0, transferBytes);
    memset(readbackBuffer, 0, transferBytes);
    for(uint32_t i = 0; i < transferBytes; ++i) {
        sourceBuffer[i] = (uint8_t)(UINT8_C(0xa5) ^ (uint8_t)(i & 0x3f));
    }
    sourceContentHash = hashBuffer(sourceBuffer, transferBytes);

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
        "same_source_across_conditions,source_pointer,source_content_hash,"
        "source_alignment_bytes,target_precondition,previous_op,previous_direction,"
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
                zeroBuffer, sourceBuffer, sourceContentHash, readbackBuffer,
                cycle, orderIndex,
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
                zeroBuffer, sourceBuffer, sourceContentHash, readbackBuffer,
                cycle, orderIndex,
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

static bool loadGroupConfig(struct GroupProbeConfig* config) {
    memset(config, 0, sizeof(*config));
    config->outputPath = getenv("BFS_GROUP_CONTEXT_PROBE_CSV");
    config->runId = getenv("BFS_GROUP_CONTEXT_PROBE_RUN_ID");
    if(config->outputPath == NULL || config->outputPath[0] == '\0') {
        return false;
    }
    if(config->runId == NULL || config->runId[0] == '\0') {
        config->runId = "bfs_group_context_probe";
    }
    if(!parseUnsigned("BFS_GROUP_CONTEXT_PROBE_PROCESS_REPEAT",
                      getenv("BFS_GROUP_CONTEXT_PROBE_PROCESS_REPEAT"), 0,
                      &config->processRepeat)
       || !parseUnsigned("BFS_GROUP_CONTEXT_PROBE_SAMPLES",
                         getenv("BFS_GROUP_CONTEXT_PROBE_SAMPLES"), 30,
                         &config->samples)
       || !parseUnsigned("BFS_GROUP_CONTEXT_PROBE_WARMUPS",
                         getenv("BFS_GROUP_CONTEXT_PROBE_WARMUPS"), 3,
                         &config->warmups)) {
        return false;
    }
    if(config->samples == 0) {
        fprintf(stderr, "Group context probe requires samples > 0\n");
        return false;
    }
    return true;
}

bool bfsGroupContextProbeRequested(void) {
    const char* outputPath = getenv("BFS_GROUP_CONTEXT_PROBE_CSV");
    return outputPath != NULL && outputPath[0] != '\0';
}

static const char* groupConditionName(enum GroupProbeCondition condition) {
    switch(condition) {
        case H2D_GROUP_THEN_H2D_GROUP:
            return "H2D_GROUP_THEN_H2D_GROUP";
        case D2H_GROUP_THEN_H2D_GROUP:
            return "D2H_GROUP_THEN_H2D_GROUP";
        case LAUNCH_D2H_GROUP_THEN_H2D_GROUP:
            return "LAUNCH_D2H_GROUP_THEN_H2D_GROUP";
        default:
            return "UNKNOWN";
    }
}

static const char* groupPredecessorChain(enum GroupProbeCondition condition) {
    switch(condition) {
        case H2D_GROUP_THEN_H2D_GROUP:
            return "H2D_GROUP>MEASURED_H2D_GROUP";
        case D2H_GROUP_THEN_H2D_GROUP:
            return "H2D_SETUP_GROUP>D2H_GROUP>MEASURED_H2D_GROUP";
        case LAUNCH_D2H_GROUP_THEN_H2D_GROUP:
            return "H2D_SETUP_GROUP>LAUNCH>D2H_GROUP>MEASURED_H2D_GROUP";
        default:
            return "UNKNOWN";
    }
}

static bool buildGroupDpuList(
    struct dpu_set_t dpuSet,
    uint32_t configuredDpus,
    const struct DPUParams* dpuParams,
    struct GroupProbeDpu* dpuList
) {
    struct dpu_set_t rank;
    struct dpu_set_t dpu;
    uint32_t rankOrdinal;
    uint32_t dpuIdInRank;
    uint32_t globalDpuId = 0;

    DPU_RANK_FOREACH(dpuSet, rank, rankOrdinal) {
        DPU_FOREACH(rank, dpu, dpuIdInRank) {
            if(globalDpuId >= configuredDpus) {
                fprintf(stderr, "Group context probe discovered too many DPUs\n");
                return false;
            }
            if(dpuParams[globalDpuId].dpuNumNodes == 0) {
                fprintf(stderr,
                        "Group context probe requires every allocated DPU to be active; "
                        "DPU %u is inactive\n",
                        globalDpuId);
                return false;
            }
            dpuList[globalDpuId].dpu = dpu;
            dpuList[globalDpuId].globalDpuId = globalDpuId;
            dpuList[globalDpuId].rankOrdinal = rankOrdinal;
            dpuList[globalDpuId].dpuIdInRank = dpuIdInRank;
            dpuList[globalDpuId].targetOffset =
                dpuParams[globalDpuId].dpuNextFrontier_m;
            ++globalDpuId;
        }
    }
    if(globalDpuId != configuredDpus) {
        fprintf(stderr,
                "Group context probe discovered %u DPUs, expected %u\n",
                globalDpuId, configuredDpus);
        return false;
    }
    return true;
}

static void copyToGroup(
    const struct GroupProbeDpu* dpuList,
    uint32_t dpuCount,
    const uint8_t* source,
    uint32_t transferBytes
) {
    for(uint32_t i = 0; i < dpuCount; ++i) {
        copyTo(dpuList[i].dpu, dpuList[i].targetOffset,
               source, transferBytes);
    }
}

static void copyFromGroupAndMerge(
    const struct GroupProbeDpu* dpuList,
    uint32_t dpuCount,
    uint8_t* readbackBuffer,
    uint64_t* mergeBuffer,
    uint32_t transferBytes
) {
    uint32_t words = transferBytes / sizeof(uint64_t);

    memset(mergeBuffer, 0, transferBytes);
    for(uint32_t i = 0; i < dpuCount; ++i) {
        copyFrom(dpuList[i].dpu, dpuList[i].targetOffset,
                 readbackBuffer, transferBytes);
        for(uint32_t word = 0; word < words; ++word) {
            mergeBuffer[word] |= ((uint64_t*)readbackBuffer)[word];
        }
    }
}

static bool verifyGroup(
    const struct GroupProbeDpu* dpuList,
    uint32_t dpuCount,
    const uint8_t* sourceBuffer,
    uint8_t* readbackBuffer,
    uint32_t transferBytes,
    uint64_t sampleIndex,
    enum GroupProbeCondition condition
) {
    for(uint32_t i = 0; i < dpuCount; ++i) {
        memset(readbackBuffer, 0, transferBytes);
        copyFrom(dpuList[i].dpu, dpuList[i].targetOffset,
                 readbackBuffer, transferBytes);
        if(memcmp(sourceBuffer, readbackBuffer, transferBytes) != 0) {
            fprintf(stderr,
                    "Group context probe readback mismatch for sample %" PRIu64
                    ", condition %s, DPU %u\n",
                    sampleIndex, groupConditionName(condition),
                    dpuList[i].globalDpuId);
            return false;
        }
    }
    return true;
}

static bool runOneGroupCondition(
    FILE* output,
    const struct GroupProbeConfig* config,
    struct dpu_set_t dpuSet,
    const struct GroupProbeDpu* dpuList,
    uint32_t configuredDpus,
    uint32_t actualRanks,
    uint32_t transferBytes,
    const uint8_t* zeroBuffer,
    const uint8_t* sourceBuffer,
    uint64_t sourceContentHash,
    uint8_t* readbackBuffer,
    uint64_t* mergeBuffer,
    uint64_t* callStartNs,
    uint64_t* callEndNs,
    uint64_t sampleIndex,
    uint32_t orderIndex,
    enum GroupProbeCondition condition,
    bool record
) {
    uint64_t predecessorStartNs;
    uint64_t predecessorEndNs;
    uint64_t launchStartNs = 0;
    uint64_t launchEndNs = 0;
    uint64_t sourcePretouchStartNs;
    uint64_t sourcePretouchEndNs;
    uint64_t groupStartNs;
    uint64_t groupEndNs;
    const char* previousOp;
    const char* previousDirection;
    unsigned int directionSwitched;
    unsigned int afterLaunch;

    copyToGroup(dpuList, configuredDpus, zeroBuffer, transferBytes);
    if(condition == H2D_GROUP_THEN_H2D_GROUP) {
        predecessorStartNs = nowNs();
        copyToGroup(dpuList, configuredDpus, zeroBuffer, transferBytes);
        predecessorEndNs = nowNs();
        previousOp = "dpu_copy_to";
        previousDirection = "TO_DPU";
        directionSwitched = 0;
        afterLaunch = 0;
    } else if(condition == D2H_GROUP_THEN_H2D_GROUP) {
        predecessorStartNs = nowNs();
        copyFromGroupAndMerge(dpuList, configuredDpus, readbackBuffer,
                              mergeBuffer, transferBytes);
        predecessorEndNs = nowNs();
        previousOp = "dpu_copy_from";
        previousDirection = "FROM_DPU";
        directionSwitched = 1;
        afterLaunch = 0;
    } else {
        launchStartNs = nowNs();
        DPU_ASSERT(dpu_launch(dpuSet, DPU_SYNCHRONOUS));
        launchEndNs = nowNs();
        predecessorStartNs = nowNs();
        copyFromGroupAndMerge(dpuList, configuredDpus, readbackBuffer,
                              mergeBuffer, transferBytes);
        predecessorEndNs = nowNs();
        previousOp = "dpu_copy_from";
        previousDirection = "FROM_DPU";
        directionSwitched = 1;
        afterLaunch = 1;
    }

    sourcePretouchStartNs = nowNs();
    touchSourceBuffer(sourceBuffer, transferBytes);
    sourcePretouchEndNs = nowNs();

    groupStartNs = nowNs();
    for(uint32_t i = 0; i < configuredDpus; ++i) {
        callStartNs[i] = nowNs();
        copyTo(dpuList[i].dpu, dpuList[i].targetOffset,
               sourceBuffer, transferBytes);
        callEndNs[i] = nowNs();
    }
    groupEndNs = nowNs();

    if(!verifyGroup(dpuList, configuredDpus, sourceBuffer, readbackBuffer,
                    transferBytes, sampleIndex, condition)) {
        return false;
    }

    if(record) {
        for(uint32_t i = 0; i < configuredDpus; ++i) {
            bool rankBoundary = i == 0
                || dpuList[i - 1].rankOrdinal != dpuList[i].rankOrdinal;
            bool sameRankAsPrevious = i > 0 && !rankBoundary;
            uint64_t nsSincePreviousSdkEvent = i == 0
                ? callStartNs[i] - predecessorEndNs
                : callStartNs[i] - callEndNs[i - 1];
            fprintf(
                output,
                "%s,%" PRIu64 ",%u,%u,%u,%" PRIu64 ",%u,%s,%s,%u,%u,%u,"
                "%u,%u,%u,%u,dpu_copy_to,TO_DPU,SINGLE_COPY,"
                "SHARED_REPLICATION,MRAM,%u,1,1,1,%u,1,CONTROLLED_GROUP_PROBE,"
                "SHARED_FIXED_BUFFER,1,0x%" PRIxPTR ",0x%016" PRIx64
                ",%u,ZERO_WRITTEN,%s,%s,%u,%u,%" PRIu64 ",%" PRIu64
                ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
                ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
                ",ok\n",
                config->runId, config->processRepeat, configuredDpus,
                NR_TASKLETS, actualRanks, sampleIndex, orderIndex,
                groupConditionName(condition), groupPredecessorChain(condition),
                i, configuredDpus, dpuList[i].globalDpuId,
                dpuList[i].rankOrdinal, dpuList[i].dpuIdInRank,
                rankBoundary ? 1U : 0U, sameRankAsPrevious ? 1U : 0U,
                transferBytes, dpuList[i].targetOffset,
                (uintptr_t)sourceBuffer, sourceContentHash,
                PROBE_HOST_BUFFER_ALIGNMENT, previousOp, previousDirection,
                directionSwitched, afterLaunch,
                predecessorEndNs - predecessorStartNs,
                launchEndNs - launchStartNs, nsSincePreviousSdkEvent,
                sourcePretouchEndNs - sourcePretouchStartNs,
                groupStartNs, groupEndNs, callStartNs[i], callEndNs[i],
                callEndNs[i] - callStartNs[i],
                groupEndNs - groupStartNs
            );
        }
    }
    return true;
}

bool bfsRunGroupContextProbe(
    struct dpu_set_t dpuSet,
    uint32_t configuredDpus,
    const struct DPUParams* dpuParams,
    uint32_t numNodes
) {
    struct GroupProbeConfig config;
    struct GroupProbeDpu* dpuList = NULL;
    uint32_t actualRanks;
    uint32_t transferBytes;
    uint8_t* zeroBuffer = NULL;
    uint8_t* sourceBuffer = NULL;
    uint8_t* readbackBuffer = NULL;
    uint64_t* mergeBuffer = NULL;
    uint64_t* callStartNs = NULL;
    uint64_t* callEndNs = NULL;
    uint64_t sourceContentHash;
    FILE* output = NULL;
    bool success = true;

    if(!loadGroupConfig(&config)) {
        return false;
    }
    if(configuredDpus == 0 || numNodes == 0 || numNodes % 64 != 0) {
        fprintf(stderr,
                "Group context probe requires DPUs and a node count divisible by 64\n");
        return false;
    }
    DPU_ASSERT(dpu_get_nr_ranks(dpuSet, &actualRanks));
    transferBytes = numNodes / 64 * sizeof(uint64_t);

    dpuList = calloc(configuredDpus, sizeof(*dpuList));
    zeroBuffer = allocateAlignedBuffer(transferBytes);
    sourceBuffer = allocateAlignedBuffer(transferBytes);
    readbackBuffer = allocateAlignedBuffer(transferBytes);
    mergeBuffer = (uint64_t*)allocateAlignedBuffer(transferBytes);
    callStartNs = calloc(configuredDpus, sizeof(*callStartNs));
    callEndNs = calloc(configuredDpus, sizeof(*callEndNs));
    if(dpuList == NULL || zeroBuffer == NULL || sourceBuffer == NULL
       || readbackBuffer == NULL || mergeBuffer == NULL
       || callStartNs == NULL || callEndNs == NULL) {
        fprintf(stderr, "Could not allocate group context probe buffers\n");
        success = false;
        goto cleanup;
    }
    if(!buildGroupDpuList(dpuSet, configuredDpus, dpuParams, dpuList)) {
        success = false;
        goto cleanup;
    }
    memset(zeroBuffer, 0, transferBytes);
    memset(readbackBuffer, 0, transferBytes);
    memset(mergeBuffer, 0, transferBytes);
    for(uint32_t i = 0; i < transferBytes; ++i) {
        sourceBuffer[i] = (uint8_t)(UINT8_C(0xa5) ^ (uint8_t)(i & 0x3f));
    }
    sourceContentHash = hashBuffer(sourceBuffer, transferBytes);

    output = fopen(config.outputPath, "w");
    if(output == NULL) {
        fprintf(stderr, "Could not open group context probe CSV %s: %s\n",
                config.outputPath, strerror(errno));
        success = false;
        goto cleanup;
    }
    fputs(
        "run_id,process_repeat,configured_dpus,num_tasklets,actual_ranks,"
        "sample_index,order_index,condition,predecessor_chain,group_index,"
        "group_size,target_global_dpu_id,rank_ordinal,dpu_id_in_rank,"
        "rank_boundary_before,same_rank_as_previous,op,direction,sdk_api_kind,"
        "logical_distribution_class,target_space,transfer_bytes_per_dpu,"
        "active_dpus,active_ranks,active_dpus_per_rank,offset_bytes,"
        "same_source_across_group,phase_class,source_buffer_class,"
        "same_source_across_conditions,source_pointer,source_content_hash,"
        "source_alignment_bytes,target_precondition,group_previous_op,"
        "group_previous_direction,direction_switched,after_launch,"
        "predecessor_group_ns,launch_ns,ns_since_previous_sdk_event,"
        "source_pretouch_ns,group_start_ns,group_end_ns,host_start_ns,"
        "host_end_ns,measured_ns,measured_group_ns,verification\n",
        output
    );

    for(uint64_t cycle = 0; cycle < config.warmups && success; ++cycle) {
        for(uint32_t orderIndex = 0;
            orderIndex < NUM_GROUP_PROBE_CONDITIONS; ++orderIndex) {
            enum GroupProbeCondition condition = (enum GroupProbeCondition)(
                (cycle + orderIndex) % NUM_GROUP_PROBE_CONDITIONS
            );
            success = runOneGroupCondition(
                output, &config, dpuSet, dpuList, configuredDpus, actualRanks,
                transferBytes, zeroBuffer, sourceBuffer, sourceContentHash,
                readbackBuffer, mergeBuffer, callStartNs, callEndNs, cycle,
                orderIndex, condition, false
            );
            if(!success) {
                break;
            }
        }
    }
    for(uint64_t cycle = 0; cycle < config.samples && success; ++cycle) {
        for(uint32_t orderIndex = 0;
            orderIndex < NUM_GROUP_PROBE_CONDITIONS; ++orderIndex) {
            enum GroupProbeCondition condition = (enum GroupProbeCondition)(
                (cycle + orderIndex) % NUM_GROUP_PROBE_CONDITIONS
            );
            success = runOneGroupCondition(
                output, &config, dpuSet, dpuList, configuredDpus, actualRanks,
                transferBytes, zeroBuffer, sourceBuffer, sourceContentHash,
                readbackBuffer, mergeBuffer, callStartNs, callEndNs, cycle,
                orderIndex, condition, true
            );
            if(!success) {
                break;
            }
        }
    }

cleanup:
    if(output != NULL && fclose(output) != 0) {
        fprintf(stderr, "Could not close group context probe CSV %s\n",
                config.outputPath);
        success = false;
    }
    free(dpuList);
    free(zeroBuffer);
    free(sourceBuffer);
    free(readbackBuffer);
    free(mergeBuffer);
    free(callStartNs);
    free(callEndNs);

    if(success) {
        printf("Group context probe wrote %" PRIu64
               " samples per condition across %u DPUs to %s\n",
               config.samples, configuredDpus, config.outputPath);
    }
    return success;
}
