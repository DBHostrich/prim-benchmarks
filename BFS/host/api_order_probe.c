#define _GNU_SOURCE

#include "api_order_probe.h"

#include <dpu_management.h>

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../support/utils.h"

#define API_ORDER_BUFFER_ALIGNMENT UINT32_C(4096)

enum ApiOrderCondition {
    CONTIGUOUS_FRONTIER_GROUP = 0,
    VISITED_FRONTIER_PARAMS_PER_DPU = 1,
    FRONTIER_PARAMS_PER_DPU = 2,
    PAIR_REVERSED_FRONTIER_PARAMS_PER_DPU = 3,
    D2H_MERGE_FRONTIER_PARAMS_PER_DPU = 4,
    NUM_API_ORDER_CONDITIONS = 5,
};

struct ApiOrderConfig {
    const char* outputPath;
    const char* runId;
    uint64_t processRepeat;
    uint64_t samples;
    uint64_t warmups;
};

struct ApiOrderDpu {
    struct dpu_set_t dpu;
    uint32_t globalDpuId;
    uint32_t rankOrdinal;
    uint32_t dpuIdInRank;
    uint32_t physicalRankId;
    uint32_t sliceId;
    uint32_t memberId;
    uint32_t frontierOffset;
    uint32_t visitedOffset;
    uint32_t paramsOffset;
};

static volatile uint64_t apiOrderTouchSink;

static uint64_t apiOrderNowNs(void) {
    struct timespec now;

    if(clock_gettime(CLOCK_MONOTONIC_RAW, &now) != 0) {
        perror("clock_gettime(CLOCK_MONOTONIC_RAW)");
        exit(EXIT_FAILURE);
    }
    return (uint64_t)now.tv_sec * UINT64_C(1000000000)
        + (uint64_t)now.tv_nsec;
}

static bool apiOrderParseUnsigned(
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

static bool loadApiOrderConfig(struct ApiOrderConfig* config) {
    memset(config, 0, sizeof(*config));
    config->outputPath = getenv("BFS_API_ORDER_PROBE_CSV");
    config->runId = getenv("BFS_API_ORDER_PROBE_RUN_ID");
    if(config->outputPath == NULL || config->outputPath[0] == '\0') {
        return false;
    }
    if(config->runId == NULL || config->runId[0] == '\0') {
        config->runId = "bfs_api_order_probe";
    }
    if(!apiOrderParseUnsigned("BFS_API_ORDER_PROBE_PROCESS_REPEAT",
                              getenv("BFS_API_ORDER_PROBE_PROCESS_REPEAT"), 0,
                              &config->processRepeat)
       || !apiOrderParseUnsigned("BFS_API_ORDER_PROBE_SAMPLES",
                                 getenv("BFS_API_ORDER_PROBE_SAMPLES"), 20,
                                 &config->samples)
       || !apiOrderParseUnsigned("BFS_API_ORDER_PROBE_WARMUPS",
                                 getenv("BFS_API_ORDER_PROBE_WARMUPS"), 3,
                                 &config->warmups)) {
        return false;
    }
    if(config->samples == 0) {
        fprintf(stderr, "API-order probe requires samples > 0\n");
        return false;
    }
    return true;
}

bool bfsApiOrderProbeRequested(void) {
    const char* outputPath = getenv("BFS_API_ORDER_PROBE_CSV");
    return outputPath != NULL && outputPath[0] != '\0';
}

static uint8_t* apiOrderAllocateAligned(uint64_t size) {
    void* buffer = NULL;
    int error = posix_memalign(&buffer, API_ORDER_BUFFER_ALIGNMENT, size);

    if(error != 0) {
        errno = error;
        return NULL;
    }
    return (uint8_t*)buffer;
}

static uint64_t apiOrderHashBuffer(const uint8_t* buffer, uint32_t size) {
    uint64_t hash = UINT64_C(14695981039346656037);

    for(uint32_t i = 0; i < size; ++i) {
        hash ^= buffer[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static void apiOrderTouchBuffer(const uint8_t* buffer, uint32_t size) {
    uint64_t checksum = 0;

    for(uint32_t offset = 0; offset < size; offset += 64) {
        checksum += buffer[offset];
    }
    checksum += buffer[size - 1];
    apiOrderTouchSink ^= checksum;
}

static void apiOrderCopyTo(
    struct dpu_set_t dpu,
    uint32_t offset,
    const uint8_t* source,
    uint32_t size
) {
    DPU_ASSERT(dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME, offset,
                           source, size));
}

static void apiOrderCopyFrom(
    struct dpu_set_t dpu,
    uint32_t offset,
    uint8_t* destination,
    uint32_t size
) {
    DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME, offset,
                             destination, size));
}

static const char* apiOrderConditionName(enum ApiOrderCondition condition) {
    switch(condition) {
        case CONTIGUOUS_FRONTIER_GROUP:
            return "CONTIGUOUS_FRONTIER_GROUP";
        case VISITED_FRONTIER_PARAMS_PER_DPU:
            return "VISITED_FRONTIER_PARAMS_PER_DPU";
        case FRONTIER_PARAMS_PER_DPU:
            return "FRONTIER_PARAMS_PER_DPU";
        case PAIR_REVERSED_FRONTIER_PARAMS_PER_DPU:
            return "PAIR_REVERSED_FRONTIER_PARAMS_PER_DPU";
        case D2H_MERGE_FRONTIER_PARAMS_PER_DPU:
            return "D2H_MERGE_FRONTIER_PARAMS_PER_DPU";
        default:
            return "UNKNOWN";
    }
}

static const char* apiOrderClass(enum ApiOrderCondition condition) {
    switch(condition) {
        case CONTIGUOUS_FRONTIER_GROUP:
            return "CONTIGUOUS_CONTROL";
        case VISITED_FRONTIER_PARAMS_PER_DPU:
            return "INIT_LIKE_ORDER";
        case FRONTIER_PARAMS_PER_DPU:
            return "PARAMS_INTERLEAVED_CONTROL";
        case PAIR_REVERSED_FRONTIER_PARAMS_PER_DPU:
            return "PAIR_REVERSED_CONTROL";
        case D2H_MERGE_FRONTIER_PARAMS_PER_DPU:
            return "ITERATIVE_LIKE_ORDER";
        default:
            return "UNKNOWN";
    }
}

static bool buildApiOrderDpuList(
    struct dpu_set_t dpuSet,
    uint32_t configuredDpus,
    const struct DPUParams* dpuParams,
    const uint32_t* dpuParamsM,
    struct ApiOrderDpu* dpuList
) {
    struct dpu_set_t rank;
    struct dpu_set_t dpu;
    struct dpu_rank_t* sdkRank;
    uint32_t rankOrdinal;
    uint32_t dpuIdInRank;
    uint32_t globalDpuId = 0;

    DPU_RANK_FOREACH(dpuSet, rank, rankOrdinal) {
        sdkRank = dpu_rank_from_set(rank);
        if(sdkRank == NULL) {
            fprintf(stderr, "API-order probe could not resolve SDK rank %u\n",
                    rankOrdinal);
            return false;
        }
        DPU_FOREACH(rank, dpu, dpuIdInRank) {
            if(globalDpuId >= configuredDpus) {
                fprintf(stderr, "API-order probe discovered too many DPUs\n");
                return false;
            }
            if(dpuParams[globalDpuId].dpuNumNodes == 0) {
                fprintf(stderr,
                        "API-order probe requires every allocated DPU to be active; "
                        "DPU %u is inactive\n",
                        globalDpuId);
                return false;
            }
            dpuList[globalDpuId].dpu = dpu;
            dpuList[globalDpuId].globalDpuId = globalDpuId;
            dpuList[globalDpuId].rankOrdinal = rankOrdinal;
            dpuList[globalDpuId].dpuIdInRank = dpuIdInRank;
            dpuList[globalDpuId].physicalRankId = dpu_get_rank_id(sdkRank);
            dpuList[globalDpuId].sliceId = dpu_get_slice_id(dpu.dpu);
            dpuList[globalDpuId].memberId = dpu_get_member_id(dpu.dpu);
            dpuList[globalDpuId].frontierOffset =
                dpuParams[globalDpuId].dpuNextFrontier_m;
            dpuList[globalDpuId].visitedOffset =
                dpuParams[globalDpuId].dpuVisited_m;
            dpuList[globalDpuId].paramsOffset = dpuParamsM[globalDpuId];
            ++globalDpuId;
        }
    }
    if(globalDpuId != configuredDpus) {
        fprintf(stderr,
                "API-order probe discovered %u DPUs, expected %u\n",
                globalDpuId, configuredDpus);
        return false;
    }
    if((configuredDpus % 2) != 0) {
        fprintf(stderr, "API-order probe requires an even DPU count\n");
        return false;
    }
    for(uint32_t i = 0; i < configuredDpus; i += 2) {
        if(dpuList[i].physicalRankId != dpuList[i + 1].physicalRankId
           || dpuList[i].sliceId != dpuList[i + 1].sliceId
           || (dpuList[i].memberId % 2) != 0
           || dpuList[i + 1].memberId != dpuList[i].memberId + 1) {
            fprintf(stderr,
                    "API-order probe DPU %u/%u are not an adjacent member pair\n",
                    i, i + 1);
            return false;
        }
    }
    return true;
}

static uint32_t apiOrderTargetIndex(
    enum ApiOrderCondition condition,
    uint32_t sequencePosition
) {
    if(condition == PAIR_REVERSED_FRONTIER_PARAMS_PER_DPU) {
        return sequencePosition ^ UINT32_C(1);
    }
    return sequencePosition;
}

static uint64_t preconditionFrontiers(
    const struct ApiOrderDpu* dpuList,
    uint32_t dpuCount,
    const uint8_t* zeroBuffer,
    uint32_t transferBytes
) {
    uint64_t endNs = 0;

    for(uint32_t i = 0; i < dpuCount; ++i) {
        apiOrderCopyTo(dpuList[i].dpu, dpuList[i].frontierOffset,
                       zeroBuffer, transferBytes);
        endNs = apiOrderNowNs();
    }
    return endNs;
}

static uint64_t readFrontierGroupAndMerge(
    const struct ApiOrderDpu* dpuList,
    uint32_t dpuCount,
    uint8_t* readbackBuffer,
    uint64_t* mergeBuffer,
    uint32_t transferBytes
) {
    uint32_t words = transferBytes / sizeof(uint64_t);
    uint64_t startNs = apiOrderNowNs();

    memset(mergeBuffer, 0, transferBytes);
    for(uint32_t i = 0; i < dpuCount; ++i) {
        apiOrderCopyFrom(dpuList[i].dpu, dpuList[i].frontierOffset,
                         readbackBuffer, transferBytes);
        for(uint32_t word = 0; word < words; ++word) {
            mergeBuffer[word] |= ((uint64_t*)readbackBuffer)[word];
        }
    }
    return apiOrderNowNs() - startNs;
}

static bool verifyApiOrderFrontiers(
    const struct ApiOrderDpu* dpuList,
    uint32_t dpuCount,
    const uint8_t* sourceBuffer,
    uint8_t* readbackBuffer,
    uint32_t transferBytes,
    uint64_t sampleIndex,
    enum ApiOrderCondition condition
) {
    for(uint32_t i = 0; i < dpuCount; ++i) {
        memset(readbackBuffer, 0, transferBytes);
        apiOrderCopyFrom(dpuList[i].dpu, dpuList[i].frontierOffset,
                         readbackBuffer, transferBytes);
        if(memcmp(sourceBuffer, readbackBuffer, transferBytes) != 0) {
            fprintf(stderr,
                    "API-order probe readback mismatch for sample %" PRIu64
                    ", condition %s, DPU %u\n",
                    sampleIndex, apiOrderConditionName(condition),
                    dpuList[i].globalDpuId);
            return false;
        }
    }
    return true;
}

static bool runOneApiOrderCondition(
    FILE* output,
    const struct ApiOrderConfig* config,
    const struct ApiOrderDpu* dpuList,
    uint32_t configuredDpus,
    uint32_t actualRanks,
    uint32_t transferBytes,
    uint32_t paramsTransferBytes,
    const uint8_t* zeroBuffer,
    const uint8_t* sourceBuffer,
    uint64_t sourceContentHash,
    const uint8_t* paramsBuffers,
    uint8_t* readbackBuffer,
    uint64_t* mergeBuffer,
    uint64_t* callStartNs,
    uint64_t* callEndNs,
    uint64_t* gapNs,
    uint64_t sampleIndex,
    uint32_t orderIndex,
    enum ApiOrderCondition condition,
    bool record
) {
    uint64_t lastSdkEndNs;
    uint64_t predecessorD2hGroupNs = 0;
    uint64_t sourcePretouchStartNs;
    uint64_t sourcePretouchEndNs;
    uint64_t sequenceStartNs;
    uint64_t sequenceEndNs;
    uint64_t frontierSumNs = 0;
    bool interleavedParams = condition != CONTIGUOUS_FRONTIER_GROUP;

    lastSdkEndNs = preconditionFrontiers(
        dpuList, configuredDpus, zeroBuffer, transferBytes
    );
    if(condition == D2H_MERGE_FRONTIER_PARAMS_PER_DPU) {
        predecessorD2hGroupNs = readFrontierGroupAndMerge(
            dpuList, configuredDpus, readbackBuffer, mergeBuffer, transferBytes
        );
        lastSdkEndNs = apiOrderNowNs();
    }

    sourcePretouchStartNs = apiOrderNowNs();
    apiOrderTouchBuffer(sourceBuffer, transferBytes);
    sourcePretouchEndNs = apiOrderNowNs();
    sequenceStartNs = apiOrderNowNs();

    for(uint32_t i = 0; i < configuredDpus; ++i) {
        uint32_t targetIndex = apiOrderTargetIndex(condition, i);

        if(condition == VISITED_FRONTIER_PARAMS_PER_DPU) {
            apiOrderCopyTo(dpuList[targetIndex].dpu,
                           dpuList[targetIndex].visitedOffset,
                           zeroBuffer, transferBytes);
            lastSdkEndNs = apiOrderNowNs();
        }

        callStartNs[targetIndex] = apiOrderNowNs();
        gapNs[targetIndex] = callStartNs[targetIndex] - lastSdkEndNs;
        apiOrderCopyTo(dpuList[targetIndex].dpu,
                       dpuList[targetIndex].frontierOffset,
                       sourceBuffer, transferBytes);
        callEndNs[targetIndex] = apiOrderNowNs();
        frontierSumNs += callEndNs[targetIndex] - callStartNs[targetIndex];
        lastSdkEndNs = callEndNs[targetIndex];

        if(interleavedParams) {
            apiOrderCopyTo(
                dpuList[targetIndex].dpu, dpuList[targetIndex].paramsOffset,
                paramsBuffers + (uint64_t)targetIndex * paramsTransferBytes,
                paramsTransferBytes
            );
            lastSdkEndNs = apiOrderNowNs();
        }
    }
    sequenceEndNs = apiOrderNowNs();

    if(!verifyApiOrderFrontiers(
           dpuList, configuredDpus, sourceBuffer, readbackBuffer,
           transferBytes, sampleIndex, condition)) {
        return false;
    }

    if(record) {
        for(uint32_t i = 0; i < configuredDpus; ++i) {
            uint32_t targetIndex = apiOrderTargetIndex(condition, i);
            uint32_t previousTarget;
            const char* previousOp;
            const char* previousEventRole;
            const char* previousDirection;
            uint32_t previousTransferBytes;
            bool directionSwitched;
            bool sameDpuAsPrevious;
            bool sameRankAsPrevious;
            bool rankBoundary = i == 0
                || dpuList[apiOrderTargetIndex(condition, i - 1)].rankOrdinal
                   != dpuList[targetIndex].rankOrdinal;

            if(condition == VISITED_FRONTIER_PARAMS_PER_DPU) {
                previousTarget = targetIndex;
                previousOp = "dpu_copy_to";
                previousEventRole = "visited_control";
                previousDirection = "TO_DPU";
                previousTransferBytes = transferBytes;
                directionSwitched = false;
            } else if(i == 0) {
                previousTarget = configuredDpus - 1;
                previousOp =
                    condition == D2H_MERGE_FRONTIER_PARAMS_PER_DPU
                    ? "dpu_copy_from" : "dpu_copy_to";
                previousEventRole =
                    condition == D2H_MERGE_FRONTIER_PARAMS_PER_DPU
                    ? "frontier_readback" : "frontier_precondition";
                previousDirection =
                    condition == D2H_MERGE_FRONTIER_PARAMS_PER_DPU
                    ? "FROM_DPU" : "TO_DPU";
                previousTransferBytes = transferBytes;
                directionSwitched =
                    condition == D2H_MERGE_FRONTIER_PARAMS_PER_DPU;
            } else if(condition == CONTIGUOUS_FRONTIER_GROUP) {
                previousTarget = apiOrderTargetIndex(condition, i - 1);
                previousOp = "dpu_copy_to";
                previousEventRole = "measured_frontier";
                previousDirection = "TO_DPU";
                previousTransferBytes = transferBytes;
                directionSwitched = false;
            } else {
                previousTarget = apiOrderTargetIndex(condition, i - 1);
                previousOp = "dpu_copy_to";
                previousEventRole = "params_control";
                previousDirection = "TO_DPU";
                previousTransferBytes = paramsTransferBytes;
                directionSwitched = false;
            }
            sameDpuAsPrevious = previousTarget == targetIndex;
            sameRankAsPrevious = dpuList[previousTarget].rankOrdinal
                == dpuList[targetIndex].rankOrdinal;

            fprintf(
                output,
                "%s,%" PRIu64 ",%u,%u,%u,%" PRIu64 ",%u,%s,%s,%u,%u,%u,"
                "%u,%u,%u,%u,%u,rank:%u/slice:%u/member:%u,%u,%u,"
                "dpu_copy_to,TO_DPU,SINGLE_COPY,"
                "SHARED_REPLICATION,MRAM,%u,1,1,1,%u,1,CONTROLLED_API_ORDER_PROBE,"
                "SHARED_FIXED_BUFFER,1,0x%" PRIxPTR ",0x%016" PRIx64
                ",%u,ZERO_WRITTEN,%s,%s,%s,%u,%u,%u,%u,%u,%u,"
                "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
                ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
                ",%" PRIu64 ",%" PRIu64 ",ok\n",
                config->runId, config->processRepeat, configuredDpus,
                NR_TASKLETS, actualRanks, sampleIndex, orderIndex,
                apiOrderConditionName(condition), apiOrderClass(condition), i,
                configuredDpus, dpuList[targetIndex].globalDpuId,
                dpuList[targetIndex].rankOrdinal,
                dpuList[targetIndex].dpuIdInRank,
                dpuList[targetIndex].physicalRankId,
                dpuList[targetIndex].sliceId,
                dpuList[targetIndex].memberId,
                dpuList[targetIndex].physicalRankId,
                dpuList[targetIndex].sliceId,
                dpuList[targetIndex].memberId,
                rankBoundary ? 1U : 0U, sameRankAsPrevious ? 1U : 0U,
                transferBytes, dpuList[targetIndex].frontierOffset,
                (uintptr_t)sourceBuffer, sourceContentHash,
                API_ORDER_BUFFER_ALIGNMENT, previousOp, previousEventRole,
                previousDirection, previousTransferBytes, previousTarget,
                sameDpuAsPrevious ? 1U : 0U,
                directionSwitched ? 1U : 0U, paramsTransferBytes,
                interleavedParams ? 1U : 0U, predecessorD2hGroupNs,
                sourcePretouchEndNs - sourcePretouchStartNs,
                gapNs[targetIndex],
                sequenceStartNs, sequenceEndNs,
                sequenceEndNs - sequenceStartNs, callStartNs[targetIndex],
                callEndNs[targetIndex],
                callEndNs[targetIndex] - callStartNs[targetIndex], frontierSumNs
            );
        }
    }
    return true;
}

bool bfsRunApiOrderProbe(
    struct dpu_set_t dpuSet,
    uint32_t configuredDpus,
    const struct DPUParams* dpuParams,
    const uint32_t* dpuParamsM,
    uint32_t numNodes
) {
    struct ApiOrderConfig config;
    struct ApiOrderDpu* dpuList = NULL;
    uint32_t actualRanks;
    uint32_t transferBytes;
    uint32_t paramsTransferBytes = ROUND_UP_TO_MULTIPLE_OF_8(
        sizeof(struct DPUParams)
    );
    uint8_t* zeroBuffer = NULL;
    uint8_t* sourceBuffer = NULL;
    uint8_t* paramsBuffers = NULL;
    uint8_t* readbackBuffer = NULL;
    uint64_t* mergeBuffer = NULL;
    uint64_t* callStartNs = NULL;
    uint64_t* callEndNs = NULL;
    uint64_t* gapNs = NULL;
    uint64_t sourceContentHash;
    FILE* output = NULL;
    bool success = true;

    if(!loadApiOrderConfig(&config)) {
        return false;
    }
    if(configuredDpus == 0 || numNodes == 0 || numNodes % 64 != 0) {
        fprintf(stderr,
                "API-order probe requires DPUs and a node count divisible by 64\n");
        return false;
    }
    DPU_ASSERT(dpu_get_nr_ranks(dpuSet, &actualRanks));
    transferBytes = numNodes / 64 * sizeof(uint64_t);

    dpuList = calloc(configuredDpus, sizeof(*dpuList));
    zeroBuffer = apiOrderAllocateAligned(transferBytes);
    sourceBuffer = apiOrderAllocateAligned(transferBytes);
    paramsBuffers = apiOrderAllocateAligned(
        (uint64_t)configuredDpus * paramsTransferBytes
    );
    readbackBuffer = apiOrderAllocateAligned(transferBytes);
    mergeBuffer = (uint64_t*)apiOrderAllocateAligned(transferBytes);
    callStartNs = calloc(configuredDpus, sizeof(*callStartNs));
    callEndNs = calloc(configuredDpus, sizeof(*callEndNs));
    gapNs = calloc(configuredDpus, sizeof(*gapNs));
    if(dpuList == NULL || zeroBuffer == NULL || sourceBuffer == NULL
       || paramsBuffers == NULL || readbackBuffer == NULL || mergeBuffer == NULL
       || callStartNs == NULL || callEndNs == NULL || gapNs == NULL) {
        fprintf(stderr, "Could not allocate API-order probe buffers\n");
        success = false;
        goto cleanup;
    }
    if(!buildApiOrderDpuList(
           dpuSet, configuredDpus, dpuParams, dpuParamsM, dpuList)) {
        success = false;
        goto cleanup;
    }

    memset(zeroBuffer, 0, transferBytes);
    memset(readbackBuffer, 0, transferBytes);
    memset(mergeBuffer, 0, transferBytes);
    memset(paramsBuffers, 0, (uint64_t)configuredDpus * paramsTransferBytes);
    for(uint32_t i = 0; i < transferBytes; ++i) {
        sourceBuffer[i] = (uint8_t)(UINT8_C(0xa5) ^ (uint8_t)(i & 0x3f));
    }
    for(uint32_t i = 0; i < configuredDpus; ++i) {
        memcpy(paramsBuffers + (uint64_t)i * paramsTransferBytes,
               &dpuParams[i], sizeof(struct DPUParams));
    }
    sourceContentHash = apiOrderHashBuffer(sourceBuffer, transferBytes);

    output = fopen(config.outputPath, "w");
    if(output == NULL) {
        fprintf(stderr, "Could not open API-order probe CSV %s: %s\n",
                config.outputPath, strerror(errno));
        success = false;
        goto cleanup;
    }
    fputs(
        "run_id,process_repeat,configured_dpus,num_tasklets,actual_ranks,"
        "sample_index,order_index,condition,api_order_class,group_index,"
        "group_size,target_global_dpu_id,rank_ordinal,dpu_id_in_rank,"
        "sdk_physical_rank_id,sdk_slice_id,sdk_member_id,"
        "physical_dpu_identity,"
        "rank_boundary_before,same_rank_as_previous_sdk_event,op,direction,"
        "sdk_api_kind,logical_distribution_class,target_space,"
        "transfer_bytes_per_dpu,active_dpus,active_ranks,"
        "active_dpus_per_rank,offset_bytes,same_source_across_group,phase_class,"
        "source_buffer_class,same_source_across_conditions,source_pointer,"
        "source_content_hash,source_alignment_bytes,target_precondition,"
        "previous_op,previous_event_role,previous_direction,"
        "previous_transfer_bytes,previous_target_global_dpu_id,"
        "same_dpu_as_previous,direction_switched,params_transfer_bytes,"
        "interleaved_params,predecessor_d2h_group_ns,source_pretouch_ns,"
        "ns_since_previous_sdk_event,sequence_start_ns,sequence_end_ns,"
        "sequence_span_ns,host_start_ns,host_end_ns,measured_ns,"
        "frontier_sum_ns,verification\n",
        output
    );

    for(uint64_t cycle = 0; cycle < config.warmups && success; ++cycle) {
        for(uint32_t orderIndex = 0;
            orderIndex < NUM_API_ORDER_CONDITIONS; ++orderIndex) {
            enum ApiOrderCondition condition = (enum ApiOrderCondition)(
                (cycle + orderIndex) % NUM_API_ORDER_CONDITIONS
            );
            success = runOneApiOrderCondition(
                output, &config, dpuList, configuredDpus, actualRanks,
                transferBytes, paramsTransferBytes, zeroBuffer, sourceBuffer,
                sourceContentHash, paramsBuffers, readbackBuffer, mergeBuffer,
                callStartNs, callEndNs, gapNs, cycle, orderIndex, condition,
                false
            );
            if(!success) {
                break;
            }
        }
    }
    for(uint64_t cycle = 0; cycle < config.samples && success; ++cycle) {
        for(uint32_t orderIndex = 0;
            orderIndex < NUM_API_ORDER_CONDITIONS; ++orderIndex) {
            enum ApiOrderCondition condition = (enum ApiOrderCondition)(
                (cycle + orderIndex) % NUM_API_ORDER_CONDITIONS
            );
            success = runOneApiOrderCondition(
                output, &config, dpuList, configuredDpus, actualRanks,
                transferBytes, paramsTransferBytes, zeroBuffer, sourceBuffer,
                sourceContentHash, paramsBuffers, readbackBuffer, mergeBuffer,
                callStartNs, callEndNs, gapNs, cycle, orderIndex, condition,
                true
            );
            if(!success) {
                break;
            }
        }
    }

cleanup:
    if(output != NULL && fclose(output) != 0) {
        fprintf(stderr, "Could not close API-order probe CSV %s\n",
                config.outputPath);
        success = false;
    }
    free(dpuList);
    free(zeroBuffer);
    free(sourceBuffer);
    free(paramsBuffers);
    free(readbackBuffer);
    free(mergeBuffer);
    free(callStartNs);
    free(callEndNs);
    free(gapNs);

    if(success) {
        printf("API-order probe wrote %" PRIu64
               " samples per condition across %u DPUs to %s\n",
               config.samples, configuredDpus, config.outputPath);
    }
    return success;
}
