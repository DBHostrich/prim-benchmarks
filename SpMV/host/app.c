#define _GNU_SOURCE

/**
* app.c
* SpMV Host Application Source File
*
*/
#include <dpu.h>
#include <dpu_log.h>
#include <dpu_management.h>

#include <assert.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "mram-management.h"
#include "../support/common.h"
#include "../support/matrix.h"
#include "../support/params.h"
#include "../support/timer.h"
#include "../support/utils.h"

#define DPU_BINARY "./bin/dpu_code"

#ifndef ENERGY
#define ENERGY 0
#endif
#if ENERGY
#include <dpu_probe.h>
#endif

struct SpmvDpuAllocation {
    bool usesPinnedRanks;
    uint32_t nrRankSets;
    struct dpu_set_t* rankSets;
    struct dpu_rank_t** combinedRanks;
};

static void destroyPinnedAllocationStorage(
    struct SpmvDpuAllocation* allocation
) {
    free(allocation->rankSets);
    free(allocation->combinedRanks);
    allocation->rankSets = NULL;
    allocation->combinedRanks = NULL;
    allocation->nrRankSets = 0;
    allocation->usesPinnedRanks = false;
}

static dpu_error_t allocateSpmvDpus(
    struct SpmvDpuAllocation* allocation,
    struct dpu_set_t* dpuSet
) {
    const char* requestedPaths = getenv("SPMV_DPU_RANK_PATHS");
    char* pathsCopy;
    char* path;
    char* pathSave = NULL;
    uint32_t expectedRanks;
    uint32_t rankIndex = 0;
    dpu_error_t status = DPU_OK;

    memset(allocation, 0, sizeof(*allocation));
    if(requestedPaths == NULL || requestedPaths[0] == '\0') {
        return dpu_alloc(NR_DPUS, NULL, dpuSet);
    }
    if(NR_DPUS % 64 != 0) {
        fprintf(stderr,
                "SPMV_DPU_RANK_PATHS requires a whole number of 64-DPU ranks\n");
        return DPU_ERR_ALLOCATION;
    }

    expectedRanks = NR_DPUS / 64;
    allocation->rankSets = calloc(expectedRanks, sizeof(*allocation->rankSets));
    allocation->combinedRanks = calloc(
        expectedRanks, sizeof(*allocation->combinedRanks)
    );
    pathsCopy = malloc(strlen(requestedPaths) + 1);
    if(allocation->rankSets == NULL || allocation->combinedRanks == NULL
       || pathsCopy == NULL) {
        free(pathsCopy);
        destroyPinnedAllocationStorage(allocation);
        return DPU_ERR_SYSTEM;
    }
    strcpy(pathsCopy, requestedPaths);

    for(path = strtok_r(pathsCopy, ",", &pathSave);
        path != NULL;
        path = strtok_r(NULL, ",", &pathSave)) {
        char profile[512];
        struct dpu_rank_t* rank;
        int profileLength;

        if(rankIndex >= expectedRanks || path[0] == '\0') {
            status = DPU_ERR_INVALID_PROFILE;
            break;
        }
        profileLength = snprintf(
            profile, sizeof(profile), "backend=hw,rankPath=%s", path
        );
        if(profileLength < 0 || (size_t)profileLength >= sizeof(profile)) {
            status = DPU_ERR_INVALID_PROFILE;
            break;
        }
        status = dpu_alloc_ranks(1, profile, &allocation->rankSets[rankIndex]);
        if(status != DPU_OK) {
            fprintf(stderr, "Could not allocate requested DPU rank %s: %s\n",
                    path, dpu_error_to_string(status));
            break;
        }
        rank = dpu_rank_from_set(allocation->rankSets[rankIndex]);
        if(rank == NULL) {
            status = DPU_ERR_INVALID_DPU_SET;
            ++rankIndex;
            break;
        }
        allocation->combinedRanks[rankIndex] = rank;
        ++rankIndex;
    }
    free(pathsCopy);

    if(status == DPU_OK && rankIndex != expectedRanks) {
        fprintf(stderr,
                "SPMV_DPU_RANK_PATHS contains %u ranks, expected %u\n",
                rankIndex, expectedRanks);
        status = DPU_ERR_ALLOCATION;
    }
    if(status != DPU_OK) {
        while(rankIndex > 0) {
            --rankIndex;
            dpu_free(allocation->rankSets[rankIndex]);
        }
        destroyPinnedAllocationStorage(allocation);
        return status;
    }

    allocation->usesPinnedRanks = true;
    allocation->nrRankSets = expectedRanks;
    if(expectedRanks == 1) {
        *dpuSet = allocation->rankSets[0];
        return DPU_OK;
    }
    memset(dpuSet, 0, sizeof(*dpuSet));
    dpuSet->kind = DPU_SET_RANKS;
    dpuSet->list.nr_ranks = expectedRanks;
    dpuSet->list.ranks = allocation->combinedRanks;
    return DPU_OK;
}

static dpu_error_t freeSpmvDpus(
    struct SpmvDpuAllocation* allocation,
    struct dpu_set_t dpuSet
) {
    dpu_error_t status = DPU_OK;
    uint32_t rankIndex;

    if(!allocation->usesPinnedRanks) {
        return dpu_free(dpuSet);
    }
    for(rankIndex = 0; rankIndex < allocation->nrRankSets; ++rankIndex) {
        dpu_error_t rankStatus = dpu_free(allocation->rankSets[rankIndex]);
        if(rankStatus != DPU_OK) {
            status = rankStatus;
        }
    }
    destroyPinnedAllocationStorage(allocation);
    return status;
}

// Main of the Host Application
int main(int argc, char** argv) {

    // Process parameters
    struct Params p = input_params(argc, argv);

    // Timing and profiling
    Timer timer;
    float loadTime = 0.0f, dpuTime = 0.0f, retrieveTime = 0.0f;
    #if ENERGY
    struct dpu_probe_t probe;
    DPU_ASSERT(dpu_probe_init("energy_probe", &probe));
    #endif

    // Allocate DPUs and load binary
    struct dpu_set_t dpu_set, dpu;
    struct SpmvDpuAllocation dpuAllocation;
    uint32_t numDPUs;
    bool traceRequested = spmvHostTraceRequested();
    uint64_t allocStartNs = 0;
    uint64_t allocEndNs = 0;
    uint64_t loadStartNs = 0;
    uint64_t loadEndNs = 0;
    if(traceRequested) {
        allocStartNs = spmvHostTraceNowNs();
    }
    DPU_ASSERT(allocateSpmvDpus(&dpuAllocation, &dpu_set));
    if(traceRequested) {
        allocEndNs = spmvHostTraceNowNs();
        loadStartNs = spmvHostTraceNowNs();
    }
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
    if(traceRequested) {
        loadEndNs = spmvHostTraceNowNs();
    }
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &numDPUs));
    PRINT_INFO(p.verbosity >= 1, "Allocated %d DPU(s)", numDPUs);
    struct SpmvHostTrace hostTrace;
    if(!spmvHostTraceInit(&hostTrace, dpu_set, numDPUs, NR_TASKLETS)) {
        DPU_ASSERT(freeSpmvDpus(&dpuAllocation, dpu_set));
        return EXIT_FAILURE;
    }
    if(spmvHostTraceEnabled(&hostTrace)) {
        spmvHostTraceRecord(&hostTrace, "dpu_alloc", "", "", false, 0,
                            0, 0, 0, NULL, allocStartNs, allocEndNs);
        spmvHostTraceRecord(&hostTrace, "dpu_load", "", "", false, 0,
                            0, 0, 0, NULL, loadStartNs, loadEndNs);
    }

    // Initialize SpMV data structures
    PRINT_INFO(p.verbosity >= 1, "Reading matrix %s", p.fileName);
    struct COOMatrix cooMatrix = readCOOMatrix(p.fileName);
    PRINT_INFO(p.verbosity >= 1, "    %u rows, %u columns, %u nonzeros", cooMatrix.numRows, cooMatrix.numCols, cooMatrix.numNonzeros);
    struct CSRMatrix csrMatrix = coo2csr(cooMatrix);
    uint32_t numRows = csrMatrix.numRows;
    uint32_t numCols = csrMatrix.numCols;
    uint32_t* rowPtrs = csrMatrix.rowPtrs;
    struct Nonzero* nonzeros = csrMatrix.nonzeros;
    float* inVector = malloc(ROUND_UP_TO_MULTIPLE_OF_8(numCols*sizeof(float)));
    initVector(inVector, numCols);
    float* outVector = malloc(ROUND_UP_TO_MULTIPLE_OF_8(numRows*sizeof(float)));

    // Partition data structure across DPUs
    uint32_t numRowsPerDPU = ROUND_UP_TO_MULTIPLE_OF_2((numRows - 1)/numDPUs + 1);
    PRINT_INFO(p.verbosity >= 1, "Assigning %u rows per DPU", numRowsPerDPU);
    struct DPUParams dpuParams[numDPUs];
    unsigned int dpuIdx = 0;
    PRINT_INFO(p.verbosity == 1, "Copying data to DPUs");
    DPU_FOREACH (dpu_set, dpu) {

        // Allocate parameters
        struct mram_heap_allocator_t allocator;
        init_allocator(&allocator);
        uint32_t dpuParams_m = mram_heap_alloc(&allocator, sizeof(struct DPUParams));

        // Find DPU's rows
        uint32_t dpuStartRowIdx = dpuIdx*numRowsPerDPU;
        uint32_t dpuNumRows;
        if(dpuStartRowIdx > numRows) {
            dpuNumRows = 0;
        } else if(dpuStartRowIdx + numRowsPerDPU > numRows) {
            dpuNumRows = numRows - dpuStartRowIdx;
        } else {
            dpuNumRows = numRowsPerDPU;
        }
        dpuParams[dpuIdx].dpuNumRows = dpuNumRows;
        PRINT_INFO(p.verbosity >= 2, "    DPU %u:", dpuIdx);
        PRINT_INFO(p.verbosity >= 2, "        Receives %u rows", dpuNumRows);

        // Partition nonzeros and copy data
        if(dpuNumRows > 0) {

            // Find DPU's CSR matrix partition
            uint32_t* dpuRowPtrs_h = &rowPtrs[dpuStartRowIdx];
            uint32_t dpuRowPtrsOffset = dpuRowPtrs_h[0];
            struct Nonzero* dpuNonzeros_h = &nonzeros[dpuRowPtrsOffset];
            uint32_t dpuNumNonzeros = dpuRowPtrs_h[dpuNumRows] - dpuRowPtrsOffset;

            // Allocate MRAM
            uint32_t dpuRowPtrs_m = mram_heap_alloc(&allocator, (dpuNumRows + 1)*sizeof(uint32_t));
            uint32_t dpuNonzeros_m = mram_heap_alloc(&allocator, dpuNumNonzeros*sizeof(struct Nonzero));
            uint32_t dpuInVector_m = mram_heap_alloc(&allocator, numCols*sizeof(float));
            uint32_t dpuOutVector_m = mram_heap_alloc(&allocator, dpuNumRows*sizeof(float));
            assert((dpuNumRows*sizeof(float))%8 == 0 && "Output sub-vector must be a multiple of 8 bytes!");
            PRINT_INFO(p.verbosity >= 2, "        Total memory allocated is %d bytes", allocator.totalAllocated);

            // Set up DPU parameters
            dpuParams[dpuIdx].dpuRowPtrsOffset = dpuRowPtrsOffset;
            dpuParams[dpuIdx].dpuRowPtrs_m = dpuRowPtrs_m;
            dpuParams[dpuIdx].dpuNonzeros_m = dpuNonzeros_m;
            dpuParams[dpuIdx].dpuInVector_m = dpuInVector_m;
            dpuParams[dpuIdx].dpuOutVector_m = dpuOutVector_m;

            // Send data to DPU
            PRINT_INFO(p.verbosity >= 2, "        Copying data to DPU");
            startTimer(&timer);
            copyToDPUTraced(&hostTrace, dpuIdx, "row_ptrs", dpu,
                            (uint8_t*)dpuRowPtrs_h, dpuRowPtrs_m,
                            (dpuNumRows + 1)*sizeof(uint32_t));
            copyToDPUTraced(&hostTrace, dpuIdx, "nonzeros", dpu,
                            (uint8_t*)dpuNonzeros_h, dpuNonzeros_m,
                            dpuNumNonzeros*sizeof(struct Nonzero));
            copyToDPUTraced(&hostTrace, dpuIdx, "input_vector", dpu,
                            (uint8_t*)inVector, dpuInVector_m, numCols*sizeof(float));
            stopTimer(&timer);
            loadTime += getElapsedTime(timer);

        }

        // Send parameters to DPU
        PRINT_INFO(p.verbosity >= 2, "        Copying parameters to DPU");
        startTimer(&timer);
        copyToDPUTraced(&hostTrace, dpuIdx, "params", dpu,
                        (uint8_t*)&dpuParams[dpuIdx], dpuParams_m,
                        sizeof(struct DPUParams));
        stopTimer(&timer);
        loadTime += getElapsedTime(timer);

        ++dpuIdx;

    }
    PRINT_INFO(p.verbosity >= 1, "    CPU-DPU Time: %f ms", loadTime*1e3);

    // Run all DPUs
    PRINT_INFO(p.verbosity >= 1, "Booting DPUs");
    startTimer(&timer);
    #if ENERGY
    DPU_ASSERT(dpu_probe_start(&probe));
    #endif
    uint64_t launchStartNs = 0;
    uint64_t launchEndNs = 0;
    if(spmvHostTraceEnabled(&hostTrace)) {
        launchStartNs = spmvHostTraceNowNs();
    }
    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
    if(spmvHostTraceEnabled(&hostTrace)) {
        launchEndNs = spmvHostTraceNowNs();
        spmvHostTraceRecord(&hostTrace, "dpu_launch", "sync", "", false, 0,
                            0, 0, 0, NULL, launchStartNs, launchEndNs);
    }
    #if ENERGY
    DPU_ASSERT(dpu_probe_stop(&probe));
    double energy;
    DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_AVERAGE, &energy));
    PRINT_INFO(p.verbosity >= 1, "    DPU Energy: %f J", energy);
    #endif
    stopTimer(&timer);
    dpuTime += getElapsedTime(timer);
    PRINT_INFO(p.verbosity >= 1, "    DPU Time: %f ms", dpuTime*1e3);

    // Copy back result
    PRINT_INFO(p.verbosity >= 1, "Copying back the result");
    startTimer(&timer);
    dpuIdx = 0;
    DPU_FOREACH (dpu_set, dpu) {
        unsigned int dpuNumRows = dpuParams[dpuIdx].dpuNumRows;
        if(dpuNumRows > 0) {
            uint32_t dpuStartRowIdx = dpuIdx*numRowsPerDPU;
            copyFromDPUTraced(&hostTrace, dpuIdx, "output_vector", dpu,
                              dpuParams[dpuIdx].dpuOutVector_m,
                              (uint8_t*)(outVector + dpuStartRowIdx),
                              dpuNumRows*sizeof(float));
        }
        ++dpuIdx;
    }
    stopTimer(&timer);
    retrieveTime += getElapsedTime(timer);
    PRINT_INFO(p.verbosity >= 1, "    DPU-CPU Time: %f ms", retrieveTime*1e3);
    if(p.verbosity == 0) PRINT("CPU-DPU Time(ms): %f    DPU Kernel Time (ms): %f    DPU-CPU Time (ms): %f", loadTime*1e3, dpuTime*1e3, retrieveTime*1e3);

    // Calculating result on CPU
    PRINT_INFO(p.verbosity >= 1, "Calculating result on CPU");
    float* outVectorReference = malloc(numRows*sizeof(float));
    for(uint32_t rowIdx = 0; rowIdx < numRows; ++rowIdx) {
        float sum = 0.0f;
        for(uint32_t i = rowPtrs[rowIdx]; i < rowPtrs[rowIdx + 1]; ++i) {
            uint32_t colIdx = nonzeros[i].col;
            float value = nonzeros[i].value;
            sum += inVector[colIdx]*value;
        }
        outVectorReference[rowIdx] = sum;
    }

    // Verify the result
    PRINT_INFO(p.verbosity >= 1, "Verifying the result");
    for(uint32_t rowIdx = 0; rowIdx < numRows; ++rowIdx) {
        float diff = (outVectorReference[rowIdx] - outVector[rowIdx])/outVectorReference[rowIdx];
        const float tolerance = 0.00001;
        if(diff > tolerance || diff < -tolerance) {
            PRINT_ERROR("Mismatch at index %u (CPU result = %f, DPU result = %f)", rowIdx, outVectorReference[rowIdx], outVector[rowIdx]);
        }
    }

    // Display DPU Logs
    if(p.verbosity >= 2) {
        PRINT_INFO(p.verbosity >= 2, "Displaying DPU Logs:");
        dpuIdx = 0;
        DPU_FOREACH (dpu_set, dpu) {
            PRINT("DPU %u:", dpuIdx);
            DPU_ASSERT(dpu_log_read(dpu, stdout));
            ++dpuIdx;
        }
    }

    // Deallocate data structures
    freeCOOMatrix(cooMatrix);
    freeCSRMatrix(csrMatrix);
    free(inVector);
    free(outVector);
    free(outVectorReference);

    uint64_t freeStartNs = 0;
    uint64_t freeEndNs = 0;
    if(spmvHostTraceEnabled(&hostTrace)) {
        freeStartNs = spmvHostTraceNowNs();
    }
    DPU_ASSERT(freeSpmvDpus(&dpuAllocation, dpu_set));
    if(spmvHostTraceEnabled(&hostTrace)) {
        freeEndNs = spmvHostTraceNowNs();
        spmvHostTraceRecord(&hostTrace, "dpu_free", "", "", false, 0,
                            0, 0, 0, NULL, freeStartNs, freeEndNs);
    }

    if(!spmvHostTraceWrite(&hostTrace)) {
        spmvHostTraceDestroy(&hostTrace);
        return EXIT_FAILURE;
    }
    spmvHostTraceDestroy(&hostTrace);

    return 0;
}
