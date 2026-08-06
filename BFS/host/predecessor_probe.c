#define _GNU_SOURCE

#include "predecessor_probe.h"

#include <dpu_management.h>

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../support/utils.h"

#define PREDECESSOR_BUFFER_ALIGNMENT UINT32_C(4096)
#define SMALL_PREDECESSOR_BYTES UINT32_C(48)

enum PredecessorCondition {
    SAME_DPU_24576B_PREDECESSOR = 0,
    SAME_DPU_48B_PREDECESSOR = 1,
    PAIRED_DPU_24576B_PREDECESSOR = 2,
    PAIRED_DPU_48B_PREDECESSOR = 3,
    NUM_PREDECESSOR_CONDITIONS = 4,
};

struct PredecessorConfig {
    const char* outputPath;
    const char* runId;
    uint64_t processRepeat;
    uint64_t samples;
    uint64_t warmups;
};

struct PredecessorDpu {
    struct dpu_set_t dpu;
    uint32_t globalDpuId;
    uint32_t rankOrdinal;
    uint32_t dpuOrdinalInRank;
    uint32_t sliceId;
    uint32_t memberId;
    uint32_t frontierOffset;
    uint32_t predecessorOffset;
    uint32_t pairedGlobalDpuId;
};

static volatile uint64_t predecessorTouchSink;

static uint64_t predecessorNowNs(void) {
    struct timespec now;

    if(clock_gettime(CLOCK_MONOTONIC_RAW, &now) != 0) {
        perror("clock_gettime(CLOCK_MONOTONIC_RAW)");
        exit(EXIT_FAILURE);
    }
    return (uint64_t)now.tv_sec * UINT64_C(1000000000)
        + (uint64_t)now.tv_nsec;
}

static bool predecessorParseUnsigned(
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

static bool loadPredecessorConfig(struct PredecessorConfig* config) {
    memset(config, 0, sizeof(*config));
    config->outputPath = getenv("BFS_PREDECESSOR_PROBE_CSV");
    config->runId = getenv("BFS_PREDECESSOR_PROBE_RUN_ID");
    if(config->outputPath == NULL || config->outputPath[0] == '\0') {
        return false;
    }
    if(config->runId == NULL || config->runId[0] == '\0') {
        config->runId = "bfs_predecessor_probe";
    }
    if(!predecessorParseUnsigned("BFS_PREDECESSOR_PROBE_PROCESS_REPEAT",
                                 getenv("BFS_PREDECESSOR_PROBE_PROCESS_REPEAT"),
                                 0, &config->processRepeat)
       || !predecessorParseUnsigned("BFS_PREDECESSOR_PROBE_SAMPLES",
                                    getenv("BFS_PREDECESSOR_PROBE_SAMPLES"),
                                    20, &config->samples)
       || !predecessorParseUnsigned("BFS_PREDECESSOR_PROBE_WARMUPS",
                                    getenv("BFS_PREDECESSOR_PROBE_WARMUPS"),
                                    3, &config->warmups)) {
        return false;
    }
    if(config->samples == 0) {
        fprintf(stderr, "Predecessor probe requires samples > 0\n");
        return false;
    }
    return true;
}

bool bfsPredecessorProbeRequested(void) {
    const char* outputPath = getenv("BFS_PREDECESSOR_PROBE_CSV");
    return outputPath != NULL && outputPath[0] != '\0';
}

static uint8_t* predecessorAllocateAligned(uint32_t size) {
    void* buffer = NULL;
    int error = posix_memalign(&buffer, PREDECESSOR_BUFFER_ALIGNMENT, size);

    if(error != 0) {
        errno = error;
        return NULL;
    }
    return (uint8_t*)buffer;
}

static uint64_t predecessorHashBuffer(const uint8_t* buffer, uint32_t size) {
    uint64_t hash = UINT64_C(14695981039346656037);

    for(uint32_t i = 0; i < size; ++i) {
        hash ^= buffer[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static void predecessorTouchBuffer(const uint8_t* buffer, uint32_t size) {
    uint64_t checksum = 0;

    for(uint32_t offset = 0; offset < size; offset += 64) {
        checksum += buffer[offset];
    }
    checksum += buffer[size - 1];
    predecessorTouchSink ^= checksum;
}

static void predecessorCopyTo(
    struct dpu_set_t dpu,
    uint32_t offset,
    const uint8_t* source,
    uint32_t size
) {
    DPU_ASSERT(dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME, offset,
                           source, size));
}

static void predecessorCopyFrom(
    struct dpu_set_t dpu,
    uint32_t offset,
    uint8_t* destination,
    uint32_t size
) {
    DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME, offset,
                             destination, size));
}

static const char* predecessorConditionName(
    enum PredecessorCondition condition
) {
    switch(condition) {
        case SAME_DPU_24576B_PREDECESSOR:
            return "SAME_DPU_24576B_PREDECESSOR";
        case SAME_DPU_48B_PREDECESSOR:
            return "SAME_DPU_48B_PREDECESSOR";
        case PAIRED_DPU_24576B_PREDECESSOR:
            return "PAIRED_DPU_24576B_PREDECESSOR";
        case PAIRED_DPU_48B_PREDECESSOR:
            return "PAIRED_DPU_48B_PREDECESSOR";
        default:
            return "UNKNOWN";
    }
}

static bool conditionUsesSameDpu(enum PredecessorCondition condition) {
    return condition == SAME_DPU_24576B_PREDECESSOR
        || condition == SAME_DPU_48B_PREDECESSOR;
}

static bool conditionUsesLargeTransfer(enum PredecessorCondition condition) {
    return condition == SAME_DPU_24576B_PREDECESSOR
        || condition == PAIRED_DPU_24576B_PREDECESSOR;
}

static uint64_t nextRandom(uint64_t* state) {
    uint64_t value = *state;

    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    *state = value;
    return value * UINT64_C(2685821657736338717);
}

static void buildTargetOrder(
    uint32_t* targetOrder,
    uint32_t configuredDpus,
    uint64_t seed
) {
    uint64_t state = seed == 0 ? UINT64_C(0x9e3779b97f4a7c15) : seed;

    for(uint32_t i = 0; i < configuredDpus; ++i) {
        targetOrder[i] = i;
    }
    for(uint32_t remaining = configuredDpus; remaining > 1; --remaining) {
        uint32_t selected = (uint32_t)(nextRandom(&state) % remaining);
        uint32_t temporary = targetOrder[remaining - 1];
        targetOrder[remaining - 1] = targetOrder[selected];
        targetOrder[selected] = temporary;
    }
}

static bool buildPredecessorDpuList(
    struct dpu_set_t dpuSet,
    uint32_t configuredDpus,
    const struct DPUParams* dpuParams,
    struct PredecessorDpu* dpuList
) {
    struct dpu_set_t rank;
    struct dpu_set_t dpu;
    uint32_t rankOrdinal;
    uint32_t dpuOrdinalInRank;
    uint32_t globalDpuId = 0;

    DPU_RANK_FOREACH(dpuSet, rank, rankOrdinal) {
        DPU_FOREACH(rank, dpu, dpuOrdinalInRank) {
            if(globalDpuId >= configuredDpus) {
                fprintf(stderr, "Predecessor probe discovered too many DPUs\n");
                return false;
            }
            if(dpuParams[globalDpuId].dpuNumNodes == 0) {
                fprintf(stderr,
                        "Predecessor probe requires every allocated DPU to be active; "
                        "DPU %u is inactive\n",
                        globalDpuId);
                return false;
            }
            dpuList[globalDpuId].dpu = dpu;
            dpuList[globalDpuId].globalDpuId = globalDpuId;
            dpuList[globalDpuId].rankOrdinal = rankOrdinal;
            dpuList[globalDpuId].dpuOrdinalInRank = dpuOrdinalInRank;
            dpuList[globalDpuId].sliceId = dpu_get_slice_id(dpu.dpu);
            dpuList[globalDpuId].memberId = dpu_get_member_id(dpu.dpu);
            dpuList[globalDpuId].frontierOffset =
                dpuParams[globalDpuId].dpuNextFrontier_m;
            dpuList[globalDpuId].predecessorOffset =
                dpuParams[globalDpuId].dpuVisited_m;
            ++globalDpuId;
        }
    }
    if(globalDpuId != configuredDpus) {
        fprintf(stderr,
                "Predecessor probe discovered %u DPUs, expected %u\n",
                globalDpuId, configuredDpus);
        return false;
    }

    for(uint32_t target = 0; target < configuredDpus; ++target) {
        uint32_t pairedOrdinal = dpuList[target].dpuOrdinalInRank ^ UINT32_C(1);
        bool found = false;
        for(uint32_t candidate = 0; candidate < configuredDpus; ++candidate) {
            if(dpuList[candidate].rankOrdinal == dpuList[target].rankOrdinal
               && dpuList[candidate].dpuOrdinalInRank == pairedOrdinal) {
                dpuList[target].pairedGlobalDpuId = candidate;
                found = true;
                break;
            }
        }
        if(!found) {
            fprintf(stderr,
                    "Predecessor probe could not pair DPU %u ordinal %u\n",
                    target, dpuList[target].dpuOrdinalInRank);
            return false;
        }
    }
    return true;
}

static const char* transitionClass(
    const struct PredecessorDpu* predecessor,
    const struct PredecessorDpu* target
) {
    if(predecessor->globalDpuId == target->globalDpuId) {
        return "SAME_DPU";
    }
    if((predecessor->dpuOrdinalInRank & UINT32_C(1)) == 0) {
        return "EVEN_TO_ODD";
    }
    return "ODD_TO_EVEN";
}

static bool runOnePredecessorCondition(
    FILE* output,
    const struct PredecessorConfig* config,
    const struct PredecessorDpu* dpuList,
    uint32_t configuredDpus,
    uint32_t actualRanks,
    uint32_t transferBytes,
    const uint8_t* zeroBuffer,
    const uint8_t* sourceBuffer,
    uint64_t sourceContentHash,
    uint8_t* readbackBuffer,
    uint64_t sampleIndex,
    uint64_t targetOrderSeed,
    uint32_t targetVisitPosition,
    uint32_t callPositionInRank,
    uint32_t conditionOrderIndex,
    enum PredecessorCondition condition,
    uint32_t targetGlobalDpuId,
    bool record
) {
    const struct PredecessorDpu* target = &dpuList[targetGlobalDpuId];
    uint32_t predecessorGlobalDpuId = conditionUsesSameDpu(condition)
        ? targetGlobalDpuId : target->pairedGlobalDpuId;
    const struct PredecessorDpu* predecessor =
        &dpuList[predecessorGlobalDpuId];
    uint32_t predecessorBytes = conditionUsesLargeTransfer(condition)
        ? transferBytes : SMALL_PREDECESSOR_BYTES;
    uint64_t preconditionStartNs;
    uint64_t preconditionEndNs;
    uint64_t sourcePretouchStartNs;
    uint64_t sourcePretouchEndNs;
    uint64_t predecessorStartNs;
    uint64_t predecessorEndNs;
    uint64_t measuredStartNs;
    uint64_t measuredEndNs;
    bool verified;

    preconditionStartNs = predecessorNowNs();
    predecessorCopyTo(target->dpu, target->frontierOffset,
                      zeroBuffer, transferBytes);
    preconditionEndNs = predecessorNowNs();

    sourcePretouchStartNs = predecessorNowNs();
    predecessorTouchBuffer(sourceBuffer, transferBytes);
    sourcePretouchEndNs = predecessorNowNs();

    predecessorStartNs = predecessorNowNs();
    predecessorCopyTo(predecessor->dpu, predecessor->predecessorOffset,
                      zeroBuffer, predecessorBytes);
    predecessorEndNs = predecessorNowNs();

    measuredStartNs = predecessorNowNs();
    predecessorCopyTo(target->dpu, target->frontierOffset,
                      sourceBuffer, transferBytes);
    measuredEndNs = predecessorNowNs();

    memset(readbackBuffer, 0, transferBytes);
    predecessorCopyFrom(target->dpu, target->frontierOffset,
                        readbackBuffer, transferBytes);
    verified = memcmp(sourceBuffer, readbackBuffer, transferBytes) == 0;
    if(!verified) {
        fprintf(stderr,
                "Predecessor probe readback mismatch for sample %" PRIu64
                ", target %u, condition %s\n",
                sampleIndex, targetGlobalDpuId,
                predecessorConditionName(condition));
        return false;
    }

    if(record) {
        fprintf(
            output,
            "%s,%" PRIu64 ",%u,%u,%u,%" PRIu64 ",%" PRIu64 ",%u,%u,%u,"
            "%s,%u,%u,%u,%u,%u,%u,%s,%u,%u,%u,%u,%u,%s,"
            "dpu_copy_to,TO_DPU,SINGLE_COPY,SHARED_REPLICATION,MRAM,"
            "%u,1,1,1,%u,1,CONTROLLED_PREDECESSOR_FACTORIAL,"
            "SHARED_FIXED_BUFFER,1,0x%" PRIxPTR ",0x%016" PRIx64
            ",%u,ZERO_WRITTEN,dpu_copy_to,TO_DPU,%u,%u,%u,%u,"
            "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
            ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
            ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",ok\n",
            config->runId, config->processRepeat, configuredDpus, NR_TASKLETS,
            actualRanks, sampleIndex, targetOrderSeed, targetVisitPosition,
            callPositionInRank, conditionOrderIndex,
            predecessorConditionName(condition), target->globalDpuId,
            target->rankOrdinal, target->dpuOrdinalInRank,
            target->dpuOrdinalInRank, target->sliceId, target->memberId,
            (target->dpuOrdinalInRank & UINT32_C(1)) == 0 ? "EVEN" : "ODD",
            predecessor->globalDpuId, predecessor->rankOrdinal,
            predecessor->dpuOrdinalInRank, predecessor->sliceId,
            predecessor->memberId, transitionClass(predecessor, target),
            transferBytes, target->frontierOffset,
            (uintptr_t)sourceBuffer, sourceContentHash,
            PREDECESSOR_BUFFER_ALIGNMENT, predecessorBytes,
            predecessor->predecessorOffset,
            predecessor->globalDpuId == target->globalDpuId ? 1U : 0U,
            predecessor->rankOrdinal == target->rankOrdinal ? 1U : 0U,
            preconditionStartNs, preconditionEndNs,
            preconditionEndNs - preconditionStartNs,
            sourcePretouchEndNs - sourcePretouchStartNs,
            predecessorStartNs, predecessorEndNs,
            predecessorEndNs - predecessorStartNs,
            measuredStartNs - predecessorEndNs, measuredStartNs,
            measuredEndNs, measuredEndNs - measuredStartNs
        );
    }
    return true;
}

static bool runPredecessorCycles(
    FILE* output,
    const struct PredecessorConfig* config,
    const struct PredecessorDpu* dpuList,
    uint32_t configuredDpus,
    uint32_t actualRanks,
    uint32_t transferBytes,
    const uint8_t* zeroBuffer,
    const uint8_t* sourceBuffer,
    uint64_t sourceContentHash,
    uint8_t* readbackBuffer,
    uint32_t* targetOrder,
    uint32_t* rankVisitCounts,
    uint64_t cycles,
    uint64_t seedBase,
    bool record
) {
    for(uint64_t cycle = 0; cycle < cycles; ++cycle) {
        uint64_t sampleIndex = cycle;
        uint64_t targetOrderSeed = seedBase
            ^ (cycle + UINT64_C(1)) * UINT64_C(0x9e3779b97f4a7c15)
            ^ (config->processRepeat + UINT64_C(1))
                * UINT64_C(0xbf58476d1ce4e5b9);
        buildTargetOrder(targetOrder, configuredDpus, targetOrderSeed);
        memset(rankVisitCounts, 0, actualRanks * sizeof(*rankVisitCounts));

        for(uint32_t position = 0; position < configuredDpus; ++position) {
            uint32_t targetGlobalDpuId = targetOrder[position];
            uint32_t rankOrdinal = dpuList[targetGlobalDpuId].rankOrdinal;
            uint32_t callPositionInRank = rankVisitCounts[rankOrdinal]++;
            for(uint32_t orderIndex = 0;
                orderIndex < NUM_PREDECESSOR_CONDITIONS; ++orderIndex) {
                enum PredecessorCondition condition =
                    (enum PredecessorCondition)(
                        (cycle + targetGlobalDpuId + orderIndex)
                        % NUM_PREDECESSOR_CONDITIONS
                    );
                if(!runOnePredecessorCondition(
                       output, config, dpuList, configuredDpus, actualRanks,
                       transferBytes, zeroBuffer, sourceBuffer,
                       sourceContentHash, readbackBuffer, sampleIndex,
                       targetOrderSeed, position, callPositionInRank, orderIndex,
                       condition, targetGlobalDpuId, record)) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool bfsRunPredecessorProbe(
    struct dpu_set_t dpuSet,
    uint32_t configuredDpus,
    const struct DPUParams* dpuParams,
    uint32_t numNodes
) {
    struct PredecessorConfig config;
    struct PredecessorDpu* dpuList = NULL;
    uint32_t actualRanks;
    uint32_t transferBytes;
    uint8_t* zeroBuffer = NULL;
    uint8_t* sourceBuffer = NULL;
    uint8_t* readbackBuffer = NULL;
    uint32_t* targetOrder = NULL;
    uint32_t* rankVisitCounts = NULL;
    uint64_t sourceContentHash;
    FILE* output = NULL;
    bool success = true;

    if(!loadPredecessorConfig(&config)) {
        return false;
    }
    if(configuredDpus == 0 || numNodes == 0 || numNodes % 64 != 0) {
        fprintf(stderr,
                "Predecessor probe requires DPUs and a node count divisible by 64\n");
        return false;
    }
    DPU_ASSERT(dpu_get_nr_ranks(dpuSet, &actualRanks));
    transferBytes = numNodes / 64 * sizeof(uint64_t);
    if(transferBytes != UINT32_C(24576)) {
        fprintf(stderr,
                "Predecessor probe requires a 24,576 B frontier, observed %u B\n",
                transferBytes);
        return false;
    }

    dpuList = calloc(configuredDpus, sizeof(*dpuList));
    zeroBuffer = predecessorAllocateAligned(transferBytes);
    sourceBuffer = predecessorAllocateAligned(transferBytes);
    readbackBuffer = predecessorAllocateAligned(transferBytes);
    targetOrder = calloc(configuredDpus, sizeof(*targetOrder));
    rankVisitCounts = calloc(actualRanks, sizeof(*rankVisitCounts));
    if(dpuList == NULL || zeroBuffer == NULL || sourceBuffer == NULL
       || readbackBuffer == NULL || targetOrder == NULL
       || rankVisitCounts == NULL) {
        fprintf(stderr, "Could not allocate predecessor probe buffers\n");
        success = false;
        goto cleanup;
    }
    if(!buildPredecessorDpuList(
           dpuSet, configuredDpus, dpuParams, dpuList)) {
        success = false;
        goto cleanup;
    }

    memset(zeroBuffer, 0, transferBytes);
    memset(readbackBuffer, 0, transferBytes);
    for(uint32_t i = 0; i < transferBytes; ++i) {
        sourceBuffer[i] = (uint8_t)(UINT8_C(0xa5) ^ (uint8_t)(i & 0x3f));
    }
    sourceContentHash = predecessorHashBuffer(sourceBuffer, transferBytes);

    output = fopen(config.outputPath, "w");
    if(output == NULL) {
        fprintf(stderr, "Could not open predecessor probe CSV %s: %s\n",
                config.outputPath, strerror(errno));
        success = false;
        goto cleanup;
    }
    fputs(
        "run_id,process_repeat,configured_dpus,num_tasklets,actual_ranks,"
        "sample_index,target_order_seed,target_visit_position,"
        "call_position_in_rank,condition_order_index,condition,"
        "target_global_dpu_id,rank_ordinal,dpu_id_in_rank,"
        "dpu_ordinal_in_rank,sdk_slice_id,sdk_member_id,ordinal_parity,"
        "predecessor_global_dpu_id,predecessor_rank_ordinal,"
        "predecessor_dpu_ordinal_in_rank,predecessor_slice_id,"
        "predecessor_member_id,transition_class,op,direction,sdk_api_kind,"
        "logical_distribution_class,target_space,transfer_bytes_per_dpu,"
        "active_dpus,active_ranks,active_dpus_per_rank,offset_bytes,"
        "same_source_across_group,phase_class,source_buffer_class,"
        "same_source_across_conditions,source_pointer,source_content_hash,"
        "source_alignment_bytes,target_precondition,previous_op,"
        "previous_direction,previous_transfer_bytes,previous_offset_bytes,"
        "same_dpu_as_previous,same_rank_as_previous,precondition_start_ns,"
        "precondition_end_ns,precondition_ns,source_pretouch_ns,"
        "predecessor_start_ns,predecessor_end_ns,predecessor_ns,"
        "ns_since_previous_sdk_event,host_start_ns,host_end_ns,measured_ns,"
        "verification\n",
        output
    );

    success = runPredecessorCycles(
        output, &config, dpuList, configuredDpus, actualRanks, transferBytes,
        zeroBuffer, sourceBuffer, sourceContentHash, readbackBuffer,
        targetOrder, rankVisitCounts, config.warmups,
        UINT64_C(0x6a09e667f3bcc909), false
    );
    if(success) {
        success = runPredecessorCycles(
            output, &config, dpuList, configuredDpus, actualRanks,
            transferBytes, zeroBuffer, sourceBuffer, sourceContentHash,
            readbackBuffer, targetOrder, rankVisitCounts, config.samples,
            UINT64_C(0xbb67ae8584caa73b), true
        );
    }

cleanup:
    if(output != NULL && fclose(output) != 0) {
        fprintf(stderr, "Could not close predecessor probe CSV %s\n",
                config.outputPath);
        success = false;
    }
    free(dpuList);
    free(zeroBuffer);
    free(sourceBuffer);
    free(readbackBuffer);
    free(targetOrder);
    free(rankVisitCounts);

    if(success) {
        printf("Predecessor probe wrote %" PRIu64
               " samples per condition across %u DPUs to %s\n",
               config.samples, configuredDpus, config.outputPath);
    }
    return success;
}
