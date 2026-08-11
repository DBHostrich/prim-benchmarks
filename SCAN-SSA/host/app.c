#define _GNU_SOURCE

/**
* app.c
* SCAN-SSA Host Application Source File
*
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <dpu.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <inttypes.h>
#include <dpu_management.h>

#include "../support/common.h"
#include "../support/timer.h"
#include "../support/params.h"
#include "host_trace.h"

// Define the DPU Binary path as DPU_BINARY here
#ifndef DPU_BINARY
#define DPU_BINARY "./bin/dpu_code"
#endif

#if ENERGY
#include <dpu_probe.h>
#endif

// Pointer declaration
static T* A;
static T* C;
static T* C2;

struct ScanDpuAllocation {
	bool uses_pinned_ranks;
	uint32_t nr_rank_sets;
	struct dpu_set_t *rank_sets;
	struct dpu_rank_t **combined_ranks;
};

static void destroy_pinned_allocation_storage(struct ScanDpuAllocation *allocation) {
	free(allocation->rank_sets);
	free(allocation->combined_ranks);
	memset(allocation, 0, sizeof(*allocation));
}

static dpu_error_t allocate_scan_dpus(
	struct ScanDpuAllocation *allocation,
	struct dpu_set_t *dpu_set
) {
	const char *requested_paths = getenv("SCAN_DPU_RANK_PATHS");
	char *paths_copy;
	char *path;
	char *path_save = NULL;
	uint32_t expected_ranks;
	uint32_t rank_index = 0;
	dpu_error_t status = DPU_OK;

	memset(allocation, 0, sizeof(*allocation));
	if (requested_paths == NULL || requested_paths[0] == '\0') {
		return dpu_alloc(NR_DPUS, NULL, dpu_set);
	}
	if (NR_DPUS % 64 != 0) {
		fprintf(stderr, "SCAN_DPU_RANK_PATHS requires complete 64-DPU ranks\n");
		return DPU_ERR_ALLOCATION;
	}

	expected_ranks = NR_DPUS / 64;
	allocation->rank_sets = calloc(expected_ranks, sizeof(*allocation->rank_sets));
	allocation->combined_ranks = calloc(
		expected_ranks, sizeof(*allocation->combined_ranks)
	);
	paths_copy = malloc(strlen(requested_paths) + 1);
	if (allocation->rank_sets == NULL || allocation->combined_ranks == NULL
		|| paths_copy == NULL) {
		free(paths_copy);
		destroy_pinned_allocation_storage(allocation);
		return DPU_ERR_SYSTEM;
	}
	strcpy(paths_copy, requested_paths);

	for (path = strtok_r(paths_copy, ",", &path_save);
		 path != NULL;
		 path = strtok_r(NULL, ",", &path_save)) {
		char profile[512];
		struct dpu_rank_t *rank;
		int profile_length;

		if (rank_index >= expected_ranks || path[0] == '\0') {
			status = DPU_ERR_INVALID_PROFILE;
			break;
		}
		profile_length = snprintf(
			profile, sizeof(profile), "backend=hw,rankPath=%s", path
		);
		if (profile_length < 0 || (size_t)profile_length >= sizeof(profile)) {
			status = DPU_ERR_INVALID_PROFILE;
			break;
		}
		status = dpu_alloc_ranks(1, profile, &allocation->rank_sets[rank_index]);
		if (status != DPU_OK) {
			fprintf(stderr, "Could not allocate requested DPU rank %s: %s\n",
				path, dpu_error_to_string(status));
			break;
		}
		rank = dpu_rank_from_set(allocation->rank_sets[rank_index]);
		if (rank == NULL) {
			status = DPU_ERR_INVALID_DPU_SET;
			++rank_index;
			break;
		}
		allocation->combined_ranks[rank_index] = rank;
		++rank_index;
	}
	free(paths_copy);

	if (status == DPU_OK && rank_index != expected_ranks) {
		fprintf(stderr, "SCAN_DPU_RANK_PATHS contains %u ranks, expected %u\n",
			rank_index, expected_ranks);
		status = DPU_ERR_ALLOCATION;
	}
	if (status != DPU_OK) {
		while (rank_index > 0) {
			--rank_index;
			dpu_free(allocation->rank_sets[rank_index]);
		}
		destroy_pinned_allocation_storage(allocation);
		return status;
	}

	allocation->uses_pinned_ranks = true;
	allocation->nr_rank_sets = expected_ranks;
	if (expected_ranks == 1) {
		*dpu_set = allocation->rank_sets[0];
		return DPU_OK;
	}
	memset(dpu_set, 0, sizeof(*dpu_set));
	dpu_set->kind = DPU_SET_RANKS;
	dpu_set->list.nr_ranks = expected_ranks;
	dpu_set->list.ranks = allocation->combined_ranks;
	return DPU_OK;
}

static dpu_error_t free_scan_dpus(
	struct ScanDpuAllocation *allocation,
	struct dpu_set_t dpu_set
) {
	dpu_error_t status = DPU_OK;
	uint32_t rank_index;

	if (!allocation->uses_pinned_ranks) {
		return dpu_free(dpu_set);
	}
	for (rank_index = 0; rank_index < allocation->nr_rank_sets; ++rank_index) {
		dpu_error_t rank_status = dpu_free(allocation->rank_sets[rank_index]);
		if (rank_status != DPU_OK) {
			status = rank_status;
		}
	}
	destroy_pinned_allocation_storage(allocation);
	return status;
}


// Create input arrays
static void read_input(T* A, unsigned int nr_elements, unsigned int nr_elements_round) {
    srand(0);
    printf("nr_elements\t%u\t", nr_elements);
    for (unsigned int i = 0; i < nr_elements; i++) {
        A[i] = (T) (rand());
    }
    for (unsigned int i = nr_elements; i < nr_elements_round; i++) {
        A[i] = 0;
    }
}

// Compute output in the host
static void scan_host(T* C, T* A, unsigned int nr_elements) {
    C[0] = A[0];
    for (unsigned int i = 1; i < nr_elements; i++) {
        C[i] = C[i - 1] + A[i];
    }
}

// Main of the Host Application
int main(int argc, char **argv) {
    struct Params p = input_params(argc, argv);
    struct dpu_set_t dpu_set, dpu;
    struct ScanDpuAllocation dpu_allocation;
    uint32_t nr_of_dpus;
    bool trace_requested = scan_host_trace_requested();
    struct ScanHostTraceMeasurement alloc_measurement = {0};
    struct ScanHostTraceMeasurement load_measurement = {0};

    if (trace_requested)
        scan_host_trace_measurement_begin(&alloc_measurement);
    DPU_ASSERT(allocate_scan_dpus(&dpu_allocation, &dpu_set));
    if (trace_requested) {
        scan_host_trace_measurement_end(&alloc_measurement);
        scan_host_trace_measurement_begin(&load_measurement);
    }
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
    if (trace_requested)
        scan_host_trace_measurement_end(&load_measurement);
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));
    printf("Allocated %u DPU(s)\n", nr_of_dpus);

#if ENERGY
    struct dpu_probe_t probe;
    DPU_ASSERT(dpu_probe_init("energy_probe", &probe));
#endif

    uint64_t input_size_64 = p.exp == 0
        ? (uint64_t)p.input_size * nr_of_dpus : p.input_size;
    if (input_size_64 == 0 || input_size_64 > UINT32_MAX) {
        fprintf(stderr, "SCAN input size is outside the supported uint32 range\n");
        DPU_ASSERT(free_scan_dpus(&dpu_allocation, dpu_set));
        return EXIT_FAILURE;
    }
    const uint32_t input_size = (uint32_t)input_size_64;
    const uint32_t input_size_dpu_raw = divceil(input_size, nr_of_dpus);
    const uint32_t input_size_dpu_round =
        input_size_dpu_raw % (NR_TASKLETS * REGS) == 0
            ? input_size_dpu_raw
            : roundup(input_size_dpu_raw, (NR_TASKLETS * REGS));
    const size_t allocated_elements =
        (size_t)input_size_dpu_round * nr_of_dpus;
    if (allocated_elements > SIZE_MAX / sizeof(T)) {
        fprintf(stderr, "SCAN host buffer size overflows size_t\n");
        DPU_ASSERT(free_scan_dpus(&dpu_allocation, dpu_set));
        return EXIT_FAILURE;
    }

    A = malloc(allocated_elements * sizeof(T));
    C = malloc(allocated_elements * sizeof(T));
    C2 = malloc(allocated_elements * sizeof(T));
    uint64_t *data_logical_bytes =
        malloc((size_t)nr_of_dpus * sizeof(*data_logical_bytes));
    if (A == NULL || C == NULL || C2 == NULL || data_logical_bytes == NULL) {
        fprintf(stderr, "Could not allocate SCAN host buffers\n");
        free(A);
        free(C);
        free(C2);
        free(data_logical_bytes);
        DPU_ASSERT(free_scan_dpus(&dpu_allocation, dpu_set));
        return EXIT_FAILURE;
    }
    for (uint32_t dpu_id = 0; dpu_id < nr_of_dpus; ++dpu_id) {
        uint64_t first = (uint64_t)dpu_id * input_size_dpu_round;
        uint64_t remaining = first < input_size ? input_size - first : 0;
        uint64_t logical_elements =
            remaining < input_size_dpu_round
                ? remaining : input_size_dpu_round;
        data_logical_bytes[dpu_id] = logical_elements * sizeof(T);
    }

    T *bufferA = A;
    T *bufferC = C2;
    read_input(A, input_size, (unsigned int)allocated_elements);

    struct ScanHostTrace host_trace;
    const char *scaling_mode = p.exp == 0 ? "WEAK" : "STRONG";
    if (!scan_host_trace_init(
            &host_trace, dpu_set, nr_of_dpus, NR_TASKLETS, input_size,
            input_size_dpu_raw, input_size_dpu_round, scaling_mode)) {
        free(A);
        free(C);
        free(C2);
        free(data_logical_bytes);
        DPU_ASSERT(free_scan_dpus(&dpu_allocation, dpu_set));
        return EXIT_FAILURE;
    }
    if (scan_host_trace_enabled(&host_trace)) {
        scan_host_trace_record_event(
            &host_trace, "dpu_alloc", "", -1, -1, &alloc_measurement);
        scan_host_trace_record_event(
            &host_trace, "dpu_load", "", -1, -1, &load_measurement);
    }

    Timer timer;
    printf("NR_TASKLETS\t%d\tBL\t%d\n", NR_TASKLETS, BL);

    for (int rep = 0; rep < p.n_warmup + p.n_reps; ++rep) {
        uint32_t i = 0;
        T accum = 0;
        int32_t warmup = rep < p.n_warmup ? 1 : 0;
        struct ScanHostTraceMeasurement operation_measurement = {0};

        if (rep >= p.n_warmup)
            start(&timer, 0, rep - p.n_warmup);
        scan_host(C, A, input_size);
        if (rep >= p.n_warmup)
            stop(&timer, 0);

        printf("Load input data\n");
        if (rep >= p.n_warmup)
            start(&timer, 1, rep - p.n_warmup);

        const uint32_t input_size_dpu = input_size_dpu_round;
        uint32_t kernel = 0;
        dpu_arguments_t input_arguments = {
            input_size_dpu * sizeof(T), kernel, 0
        };
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, &input_arguments));
        }
        if (scan_host_trace_enabled(&host_trace))
            scan_host_trace_measurement_begin(&operation_measurement);
        DPU_ASSERT(dpu_push_xfer(
            dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0,
            sizeof(input_arguments), DPU_XFER_DEFAULT));
        if (scan_host_trace_enabled(&host_trace)) {
            scan_host_trace_measurement_end(&operation_measurement);
            scan_host_trace_record_transfer(
                &host_trace, "input_arguments_scan", "TO_DPU", rep, warmup,
                "WRAM", "DPU_INPUT_ARGUMENTS", 0, NULL,
                sizeof(input_arguments), sizeof(input_arguments),
                rep, rep, "NONE", 0, 0, &operation_measurement);
        }

        i = 0;
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(
                dpu, bufferA + (size_t)input_size_dpu * i));
        }
        if (scan_host_trace_enabled(&host_trace))
            scan_host_trace_measurement_begin(&operation_measurement);
        DPU_ASSERT(dpu_push_xfer(
            dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0,
            (size_t)input_size_dpu * sizeof(T), DPU_XFER_DEFAULT));
        if (scan_host_trace_enabled(&host_trace)) {
            scan_host_trace_measurement_end(&operation_measurement);
            scan_host_trace_record_transfer(
                &host_trace, "input_data", "TO_DPU", rep, warmup,
                "MRAM", "DPU_MRAM_HEAP_POINTER_NAME", 0,
                data_logical_bytes, 0,
                (uint64_t)input_size_dpu * sizeof(T),
                rep, rep, "NONE", 1, 0, &operation_measurement);
        }
        if (rep >= p.n_warmup)
            stop(&timer, 1);

        printf("Run program on DPU(s)\n");
        if (rep >= p.n_warmup) {
            start(&timer, 2, rep - p.n_warmup);
#if ENERGY
            DPU_ASSERT(dpu_probe_start(&probe));
#endif
        }
        if (scan_host_trace_enabled(&host_trace))
            scan_host_trace_measurement_begin(&operation_measurement);
        DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
        if (scan_host_trace_enabled(&host_trace)) {
            scan_host_trace_measurement_end(&operation_measurement);
            scan_host_trace_record_event(
                &host_trace, "dpu_launch", "scan_sync", rep, warmup,
                &operation_measurement);
        }
        if (rep >= p.n_warmup) {
            stop(&timer, 2);
#if ENERGY
            DPU_ASSERT(dpu_probe_stop(&probe));
#endif
        }

#if PRINT
        {
            unsigned int each_dpu = 0;
            printf("Display DPU Logs\n");
            DPU_FOREACH (dpu_set, dpu) {
                printf("DPU#%d:\n", each_dpu);
                DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
                each_dpu++;
            }
        }
#endif

        printf("Retrieve partial results\n");
        dpu_results_t *results =
            malloc((size_t)nr_of_dpus * sizeof(*results));
        T *results_scan = malloc((size_t)nr_of_dpus * sizeof(*results_scan));
        dpu_results_t **results_retrieve =
            calloc(nr_of_dpus, sizeof(*results_retrieve));
        if (results == NULL || results_scan == NULL || results_retrieve == NULL) {
            fprintf(stderr, "Could not allocate SCAN partial-result buffers\n");
            free(results);
            free(results_scan);
            free(results_retrieve);
            scan_host_trace_destroy(&host_trace);
            free(A);
            free(C);
            free(C2);
            free(data_logical_bytes);
            DPU_ASSERT(free_scan_dpus(&dpu_allocation, dpu_set));
            return EXIT_FAILURE;
        }

        if (rep >= p.n_warmup)
            start(&timer, 3, rep - p.n_warmup);
        i = 0;
        DPU_FOREACH(dpu_set, dpu, i) {
            results_retrieve[i] =
                malloc(NR_TASKLETS * sizeof(dpu_results_t));
            if (results_retrieve[i] == NULL) {
                fprintf(stderr, "Could not allocate a SCAN DPU result buffer\n");
                exit(EXIT_FAILURE);
            }
            DPU_ASSERT(dpu_prepare_xfer(dpu, results_retrieve[i]));
        }
        if (scan_host_trace_enabled(&host_trace))
            scan_host_trace_measurement_begin(&operation_measurement);
        DPU_ASSERT(dpu_push_xfer(
            dpu_set, DPU_XFER_FROM_DPU, "DPU_RESULTS", 0,
            NR_TASKLETS * sizeof(dpu_results_t), DPU_XFER_DEFAULT));
        if (scan_host_trace_enabled(&host_trace)) {
            scan_host_trace_measurement_end(&operation_measurement);
            scan_host_trace_record_transfer(
                &host_trace, "partial_results", "FROM_DPU", rep, warmup,
                "WRAM", "DPU_RESULTS", 0, NULL,
                NR_TASKLETS * sizeof(dpu_results_t),
                NR_TASKLETS * sizeof(dpu_results_t),
                rep, rep, "NONE", 0, 0, &operation_measurement);
        }

        for (i = 0; i < nr_of_dpus; ++i) {
            results[i].t_count =
                results_retrieve[i][NR_TASKLETS - 1].t_count;
            free(results_retrieve[i]);
            T temp = results[i].t_count;
            results_scan[i] = accum;
            accum += temp;
        }

        kernel = 1;
        dpu_arguments_t *input_arguments_2 =
            malloc((size_t)nr_of_dpus * sizeof(*input_arguments_2));
        if (input_arguments_2 == NULL) {
            fprintf(stderr, "Could not allocate SCAN add arguments\n");
            exit(EXIT_FAILURE);
        }
        for (i = 0; i < nr_of_dpus; ++i) {
            input_arguments_2[i].size = input_size_dpu * sizeof(T);
            input_arguments_2[i].kernel = kernel;
            input_arguments_2[i].t_count = results_scan[i];
        }
        i = 0;
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, &input_arguments_2[i]));
        }
        if (scan_host_trace_enabled(&host_trace))
            scan_host_trace_measurement_begin(&operation_measurement);
        DPU_ASSERT(dpu_push_xfer(
            dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0,
            sizeof(input_arguments_2[0]), DPU_XFER_DEFAULT));
        if (scan_host_trace_enabled(&host_trace)) {
            scan_host_trace_measurement_end(&operation_measurement);
            scan_host_trace_record_transfer(
                &host_trace, "input_arguments_add", "TO_DPU", rep, warmup,
                "WRAM", "DPU_INPUT_ARGUMENTS", 0, NULL,
                sizeof(input_arguments_2[0]), sizeof(input_arguments_2[0]),
                rep, rep, "NONE", 0, 0, &operation_measurement);
        }
        if (rep >= p.n_warmup)
            stop(&timer, 3);

        printf("Run add program on DPU(s)\n");
        if (rep >= p.n_warmup) {
            start(&timer, 4, rep - p.n_warmup);
#if ENERGY
            DPU_ASSERT(dpu_probe_start(&probe));
#endif
        }
        if (scan_host_trace_enabled(&host_trace))
            scan_host_trace_measurement_begin(&operation_measurement);
        DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
        if (scan_host_trace_enabled(&host_trace)) {
            scan_host_trace_measurement_end(&operation_measurement);
            scan_host_trace_record_event(
                &host_trace, "dpu_launch", "add_sync", rep, warmup,
                &operation_measurement);
        }
        if (rep >= p.n_warmup) {
            stop(&timer, 4);
#if ENERGY
            DPU_ASSERT(dpu_probe_stop(&probe));
#endif
        }

        printf("Retrieve results\n");
        if (rep >= p.n_warmup)
            start(&timer, 5, rep - p.n_warmup);
        i = 0;
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(
                dpu, bufferC + (size_t)input_size_dpu * i));
        }
        if (scan_host_trace_enabled(&host_trace))
            scan_host_trace_measurement_begin(&operation_measurement);
        DPU_ASSERT(dpu_push_xfer(
            dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME,
            (size_t)input_size_dpu * sizeof(T),
            (size_t)input_size_dpu * sizeof(T), DPU_XFER_DEFAULT));
        if (scan_host_trace_enabled(&host_trace)) {
            scan_host_trace_measurement_end(&operation_measurement);
            scan_host_trace_record_transfer(
                &host_trace, "output_data", "FROM_DPU", rep, warmup,
                "MRAM", "DPU_MRAM_HEAP_POINTER_NAME",
                (uint64_t)input_size_dpu * sizeof(T),
                data_logical_bytes, 0,
                (uint64_t)input_size_dpu * sizeof(T),
                rep, rep, "NONE", 1, 0, &operation_measurement);
        }
        if (rep >= p.n_warmup)
            stop(&timer, 5);

        free(results);
        free(results_scan);
        free(results_retrieve);
        free(input_arguments_2);
    }

    printf("CPU ");
    print(&timer, 0, p.n_reps);
    printf("CPU-DPU ");
    print(&timer, 1, p.n_reps);
    printf("DPU Kernel Scan ");
    print(&timer, 2, p.n_reps);
    printf("Inter-DPU (Scan) ");
    print(&timer, 3, p.n_reps);
    printf("DPU Kernel Add ");
    print(&timer, 4, p.n_reps);
    printf("DPU-CPU ");
    print(&timer, 5, p.n_reps);

#if ENERGY
    double energy;
    DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_AVERAGE, &energy));
    printf("DPU Energy (J): %f\t", energy);
#endif

    bool status = true;
    for (uint32_t i = 0; i < input_size; ++i) {
        if (C[i] != bufferC[i]) {
            status = false;
#if PRINT
            printf("%u: %" PRId64 " -- %" PRId64 "\n",
                i, (int64_t)C[i], (int64_t)bufferC[i]);
#endif
        }
    }
    if (status)
        printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
    else
        printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");

    free(A);
    free(C);
    free(C2);
    free(data_logical_bytes);

    struct ScanHostTraceMeasurement free_measurement = {0};
    if (scan_host_trace_enabled(&host_trace))
        scan_host_trace_measurement_begin(&free_measurement);
    DPU_ASSERT(free_scan_dpus(&dpu_allocation, dpu_set));
    if (scan_host_trace_enabled(&host_trace)) {
        scan_host_trace_measurement_end(&free_measurement);
        scan_host_trace_record_event(
            &host_trace, "dpu_free", "", -1, -1, &free_measurement);
    }
    if (!scan_host_trace_write(&host_trace)) {
        scan_host_trace_destroy(&host_trace);
        return EXIT_FAILURE;
    }
    scan_host_trace_destroy(&host_trace);

#if ENERGY
    DPU_ASSERT(dpu_probe_deinit(&probe));
#endif

    return status ? EXIT_SUCCESS : EXIT_FAILURE;
}
