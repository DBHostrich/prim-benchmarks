/**
 * app.c
 * GEMV Host Application Source File
 *
 */

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <dpu.h>
#include <dpu_log.h>
#include <dpu_management.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>

#if ENERGY
#include <dpu_probe.h>
#endif

#include "../support/common.h"
#include "../support/timer.h"
#include "../support/params.h"
#include "host_trace.h"

// Define the DPU Binary path as DPU_BINARY here
#ifndef DPU_BINARY
#define DPU_BINARY "./bin/gemv_dpu"
#endif

static T* A;
static T* B;
static T* C;
static T* C_dpu;

struct GemvDpuAllocation {
	bool uses_pinned_ranks;
	uint32_t nr_rank_sets;
	struct dpu_set_t *rank_sets;
	struct dpu_rank_t **combined_ranks;
};

static void destroy_pinned_allocation_storage(struct GemvDpuAllocation *allocation) {
	free(allocation->rank_sets);
	free(allocation->combined_ranks);
	memset(allocation, 0, sizeof(*allocation));
}

static dpu_error_t allocate_gemv_dpus(
	struct GemvDpuAllocation *allocation,
	struct dpu_set_t *dpu_set
) {
	const char *requested_paths = getenv("GEMV_DPU_RANK_PATHS");
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
		fprintf(stderr, "GEMV_DPU_RANK_PATHS requires complete 64-DPU ranks\n");
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
		fprintf(stderr, "GEMV_DPU_RANK_PATHS contains %u ranks, expected %u\n",
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

static dpu_error_t free_gemv_dpus(
	struct GemvDpuAllocation *allocation,
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
static void init_data(T* A, T* B, unsigned int m_size, unsigned int n_size) {
	srand(0);

	for (unsigned int i = 0; i < m_size * n_size; i++)
	{
		A[i] = (unsigned int) (rand()%50);
	}

	for (unsigned int i = 0; i < n_size; i++)
	{
		B[i] = (unsigned int) (rand()%50);
	}
}

// Compute output in the host
static void gemv_host(T* C, T* A, T* B, unsigned int m_size, unsigned int n_size) {
	for (unsigned int i = 0; i < m_size; i++)
	{
		C[i] = 0;
	}

	for (unsigned int m = 0; m < m_size; m++) {
		for (unsigned int n = 0; n < n_size; n++)
		{
			C[m] += A[m * n_size + n] * B[n];
		}
	}
}

enum GemvTransferOrder {
	GEMV_MATRIX_THEN_VECTOR,
	GEMV_VECTOR_THEN_MATRIX,
};

enum GemvVectorReplayMode {
	GEMV_VECTOR_REPLAY_NONE,
	GEMV_VECTOR_REPLAY_IDENTICAL,
};

static bool parse_transfer_order(enum GemvTransferOrder *order) {
	const char *value = getenv("GEMV_TRANSFER_ORDER");
	if (value == NULL || value[0] == '\0'
		|| strcmp(value, "MATRIX_THEN_VECTOR") == 0) {
		*order = GEMV_MATRIX_THEN_VECTOR;
		return true;
	}
	if (strcmp(value, "VECTOR_THEN_MATRIX") == 0) {
		*order = GEMV_VECTOR_THEN_MATRIX;
		return true;
	}
	fprintf(stderr, "Unsupported GEMV_TRANSFER_ORDER: %s\n", value);
	return false;
}

static bool parse_vector_replay_mode(enum GemvVectorReplayMode *mode) {
	const char *value = getenv("GEMV_VECTOR_REPLAY_MODE");
	if (value == NULL || value[0] == '\0' || strcmp(value, "NONE") == 0) {
		*mode = GEMV_VECTOR_REPLAY_NONE;
		return true;
	}
	if (strcmp(value, "IDENTICAL_REPLAY") == 0) {
		*mode = GEMV_VECTOR_REPLAY_IDENTICAL;
		return true;
	}
	fprintf(stderr, "Unsupported GEMV_VECTOR_REPLAY_MODE: %s\n", value);
	return false;
}

static void push_input_matrix(
	struct dpu_set_t dpu_set,
	unsigned int n_size,
	uint32_t n_size_pad,
	uint32_t max_rows_per_dpu,
	const uint64_t *matrix_logical_bytes,
	unsigned int rep,
	int32_t warmup,
	uint64_t mram_push_ordinal_since_launch,
	struct GemvHostTrace *host_trace
) {
	struct dpu_set_t dpu;
	uint32_t i = 0;
	struct GemvHostTraceMeasurement measurement = {0};
	DPU_FOREACH(dpu_set, dpu, i) {
		DPU_ASSERT(dpu_prepare_xfer(
			dpu, A + dpu_info[i].prev_rows_dpu * n_size
		));
	}
	if (gemv_host_trace_enabled(host_trace))
		gemv_host_trace_measurement_begin(&measurement);
	DPU_ASSERT(dpu_push_xfer(
		dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0,
		max_rows_per_dpu * n_size_pad * sizeof(T), DPU_XFER_DEFAULT
	));
	if (gemv_host_trace_enabled(host_trace)) {
		gemv_host_trace_measurement_end(&measurement);
		gemv_host_trace_record_transfer(
			host_trace, "input_matrix", "TO_DPU", rep, warmup,
			"MRAM", "DPU_MRAM_HEAP_POINTER_NAME", 0,
			matrix_logical_bytes, 0,
			(uint64_t)max_rows_per_dpu * n_size_pad * sizeof(T),
			rep, rep, "NONE", mram_push_ordinal_since_launch, &measurement
		);
	}
}

static void push_input_vector(
	struct dpu_set_t dpu_set,
	uint32_t n_size,
	uint32_t n_size_pad,
	uint32_t max_rows_per_dpu,
	unsigned int rep,
	int32_t warmup,
	uint64_t source_buffer_use_count_before,
	uint64_t target_region_access_count_before,
	const char *diagnostic_copy_ordinal,
	uint64_t mram_push_ordinal_since_launch,
	struct GemvHostTrace *host_trace
) {
	struct dpu_set_t dpu;
	struct GemvHostTraceMeasurement measurement = {0};
	DPU_FOREACH(dpu_set, dpu) {
		DPU_ASSERT(dpu_prepare_xfer(dpu, B));
	}
	if (gemv_host_trace_enabled(host_trace))
		gemv_host_trace_measurement_begin(&measurement);
	DPU_ASSERT(dpu_push_xfer(
		dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME,
		(uint64_t)max_rows_per_dpu * n_size_pad * sizeof(T),
		n_size_pad * sizeof(T), DPU_XFER_DEFAULT
	));
	if (gemv_host_trace_enabled(host_trace)) {
		gemv_host_trace_measurement_end(&measurement);
		gemv_host_trace_record_transfer(
			host_trace, "input_vector", "TO_DPU", rep, warmup,
			"MRAM", "DPU_MRAM_HEAP_POINTER_NAME",
			(uint64_t)max_rows_per_dpu * n_size_pad * sizeof(T),
			NULL, (uint64_t)n_size * sizeof(T),
			(uint64_t)n_size_pad * sizeof(T),
			source_buffer_use_count_before,
			target_region_access_count_before,
			diagnostic_copy_ordinal, mram_push_ordinal_since_launch,
			&measurement
		);
	}
}

// Main of the Host Application
int main(int argc, char **argv) {

	struct Params p = input_params(argc, argv);
	enum GemvTransferOrder transfer_order;
	enum GemvVectorReplayMode vector_replay_mode;
	if (!parse_transfer_order(&transfer_order))
		return EXIT_FAILURE;
	if (!parse_vector_replay_mode(&vector_replay_mode))
		return EXIT_FAILURE;
	if (vector_replay_mode == GEMV_VECTOR_REPLAY_IDENTICAL
		&& transfer_order != GEMV_MATRIX_THEN_VECTOR) {
		fprintf(stderr,
			"IDENTICAL_REPLAY requires MATRIX_THEN_VECTOR order\n");
		return EXIT_FAILURE;
	}

	struct dpu_set_t dpu_set, dpu;
	struct GemvDpuAllocation dpu_allocation;
	uint32_t nr_of_dpus;
	bool trace_requested = gemv_host_trace_requested();
	struct GemvHostTraceMeasurement alloc_measurement = {0};
	struct GemvHostTraceMeasurement load_measurement = {0};

	// Allocate DPUs and load binary
	if (trace_requested)
		gemv_host_trace_measurement_begin(&alloc_measurement);
	DPU_ASSERT(allocate_gemv_dpus(&dpu_allocation, &dpu_set));
	if (trace_requested) {
		gemv_host_trace_measurement_end(&alloc_measurement);
		gemv_host_trace_measurement_begin(&load_measurement);
	}
	DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
	if (trace_requested)
		gemv_host_trace_measurement_end(&load_measurement);
	DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));

#if ENERGY
	struct dpu_probe_t probe;
	DPU_ASSERT(dpu_probe_init("energy_probe", &probe));
#endif

	unsigned int i;
	unsigned int m_size = p.m_size;
	unsigned int n_size = p.n_size;

	// Initialize help data
	dpu_info = (struct dpu_info_t *) malloc(nr_of_dpus * sizeof(struct dpu_info_t));
	dpu_arguments_t *input_args = (dpu_arguments_t *) malloc(nr_of_dpus * sizeof(dpu_arguments_t));
	if (dpu_info == NULL || input_args == NULL) {
		fprintf(stderr, "Could not allocate GEMV metadata buffers\n");
		free(dpu_info);
		free(input_args);
		DPU_ASSERT(free_gemv_dpus(&dpu_allocation, dpu_set));
		return EXIT_FAILURE;
	}
	uint32_t max_rows_per_dpu = 0;
	uint32_t n_size_pad = n_size;
	if(n_size % 2 == 1)
	{
		n_size_pad++;
	}

	i = 0;
	DPU_FOREACH(dpu_set, dpu, i) {
		uint32_t rows_per_dpu;
		uint32_t prev_rows_dpu = 0;
		uint32_t chunks = m_size / nr_of_dpus;
		rows_per_dpu = chunks;
		uint32_t rest_rows = m_size % nr_of_dpus;
		if (i < rest_rows)
			rows_per_dpu++;
		if (rest_rows > 0) {
			if (i >= rest_rows)
				prev_rows_dpu = rest_rows * (chunks + 1) + (i - rest_rows) * chunks;
			else
				prev_rows_dpu = i * (chunks + 1);
		} else {
			prev_rows_dpu = i * chunks;
		}

		// Keep max rows for parallel transfers
		uint32_t rows_per_dpu_pad = rows_per_dpu;
		if (rows_per_dpu_pad % 2 == 1) // 4-byte elements
			rows_per_dpu_pad++;
		if (rows_per_dpu_pad > max_rows_per_dpu)
			max_rows_per_dpu = rows_per_dpu_pad;

		dpu_info[i].rows_per_dpu = rows_per_dpu;
		dpu_info[i].rows_per_dpu_pad = rows_per_dpu_pad;
		dpu_info[i].prev_rows_dpu = prev_rows_dpu;

		// Copy input arguments to DPU
		input_args[i].n_size = n_size;
		input_args[i].n_size_pad = n_size_pad;
		input_args[i].nr_rows = rows_per_dpu;
	}

	A = malloc(max_rows_per_dpu * nr_of_dpus * n_size_pad * sizeof(T));
	B = malloc(n_size_pad * sizeof(T));
	C = malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T));
	C_dpu = malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T));
	uint64_t *matrix_logical_bytes = malloc(nr_of_dpus * sizeof(uint64_t));
	uint64_t *result_logical_bytes = malloc(nr_of_dpus * sizeof(uint64_t));
	if (A == NULL || B == NULL || C == NULL || C_dpu == NULL
		|| matrix_logical_bytes == NULL || result_logical_bytes == NULL) {
		fprintf(stderr, "Could not allocate GEMV host buffers\n");
		free(A);
		free(B);
		free(C);
		free(C_dpu);
		free(matrix_logical_bytes);
		free(result_logical_bytes);
		free(dpu_info);
		free(input_args);
		DPU_ASSERT(free_gemv_dpus(&dpu_allocation, dpu_set));
		return EXIT_FAILURE;
	}
	for (i = 0; i < nr_of_dpus; ++i) {
		matrix_logical_bytes[i] =
			(uint64_t)dpu_info[i].rows_per_dpu * n_size * sizeof(T);
		result_logical_bytes[i] =
			(uint64_t)dpu_info[i].rows_per_dpu * sizeof(T);
	}

	struct GemvHostTrace host_trace;
	if (!gemv_host_trace_init(
			&host_trace, dpu_set, nr_of_dpus, NR_TASKLETS,
			m_size, n_size, n_size_pad, max_rows_per_dpu)) {
		free(A);
		free(B);
		free(C);
		free(C_dpu);
		free(matrix_logical_bytes);
		free(result_logical_bytes);
		free(dpu_info);
		free(input_args);
		DPU_ASSERT(free_gemv_dpus(&dpu_allocation, dpu_set));
		return EXIT_FAILURE;
	}
	if (gemv_host_trace_enabled(&host_trace)) {
		gemv_host_trace_record_event(
			&host_trace, "dpu_alloc", "", -1, -1,
			&alloc_measurement
		);
		gemv_host_trace_record_event(
			&host_trace, "dpu_load", "", -1, -1,
			&load_measurement
		);
	}

	// Initialize data with arbitrary data
	init_data(A, B, m_size, n_size);

	// Timer
	Timer timer;

	// Compute output on CPU (performance comparison and verification purposes)
	start(&timer, 0, 0);
	gemv_host(C, A, B, m_size, n_size);
	stop(&timer, 0);
	for (unsigned int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {
		int32_t warmup = rep < p.n_warmup ? 1 : 0;
		uint64_t vector_use_count =
			vector_replay_mode == GEMV_VECTOR_REPLAY_IDENTICAL
			? (uint64_t)rep * 2u : rep;
		struct GemvHostTraceMeasurement operation_measurement = {0};

		if (rep >= p.n_warmup)
			start(&timer, 1, rep - p.n_warmup);
		// Input arguments
		i = 0;
		DPU_FOREACH(dpu_set, dpu, i) {
			// Copy input arguments to DPU
			input_args[i].max_rows = max_rows_per_dpu;

			DPU_ASSERT(dpu_prepare_xfer(dpu, input_args + i));
		}

		if (gemv_host_trace_enabled(&host_trace))
			gemv_host_trace_measurement_begin(&operation_measurement);
		DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));
		if (gemv_host_trace_enabled(&host_trace)) {
			gemv_host_trace_measurement_end(&operation_measurement);
			gemv_host_trace_record_transfer(
				&host_trace, "input_arguments", "TO_DPU", rep, warmup,
				"WRAM", "DPU_INPUT_ARGUMENTS", 0, NULL,
				sizeof(dpu_arguments_t), sizeof(dpu_arguments_t),
				rep, rep, "NONE", 0, &operation_measurement
			);
		}

		// Copy input array and vector using the selected control order.
		if (transfer_order == GEMV_VECTOR_THEN_MATRIX) {
			push_input_vector(
				dpu_set, n_size, n_size_pad, max_rows_per_dpu,
				rep, warmup, rep, rep, "PRIMARY", 1, &host_trace
			);
			push_input_matrix(
				dpu_set, n_size, n_size_pad, max_rows_per_dpu,
				matrix_logical_bytes, rep, warmup, 2, &host_trace
			);
		} else {
			push_input_matrix(
				dpu_set, n_size, n_size_pad, max_rows_per_dpu,
				matrix_logical_bytes, rep, warmup, 1, &host_trace
			);
			push_input_vector(
				dpu_set, n_size, n_size_pad, max_rows_per_dpu,
				rep, warmup, vector_use_count, vector_use_count,
				"PRIMARY", 2, &host_trace
			);
			if (vector_replay_mode == GEMV_VECTOR_REPLAY_IDENTICAL) {
				push_input_vector(
					dpu_set, n_size, n_size_pad, max_rows_per_dpu,
					rep, warmup, vector_use_count + 1u,
					vector_use_count + 1u, "IDENTICAL_REPLAY", 3,
					&host_trace
				);
			}
		}

		if (rep >= p.n_warmup)
			stop(&timer, 1);

		// Run kernel on DPUs
		if (rep >= p.n_warmup)
		{
			start(&timer, 2, rep - p.n_warmup);
#if ENERGY
			DPU_ASSERT(dpu_probe_start(&probe));
#endif
		}

		if (gemv_host_trace_enabled(&host_trace))
			gemv_host_trace_measurement_begin(&operation_measurement);
		DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
		if (gemv_host_trace_enabled(&host_trace)) {
			gemv_host_trace_measurement_end(&operation_measurement);
			gemv_host_trace_record_event(
				&host_trace, "dpu_launch", "sync", rep, warmup,
				&operation_measurement
			);
		}

		if (rep >= p.n_warmup)
		{
			stop(&timer, 2);
#if ENERGY
			DPU_ASSERT(dpu_probe_stop(&probe));
#endif
		}
#if PRINT
		// Display DPU Logs
		DPU_FOREACH(dpu_set, dpu) {
			DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
		}
#endif

		// Retrieve results
		if (rep >= p.n_warmup)
			start(&timer, 3, rep - p.n_warmup);
		i = 0;
		DPU_FOREACH(dpu_set, dpu, i) {
			DPU_ASSERT(dpu_prepare_xfer(dpu, C_dpu + i * max_rows_per_dpu));
		}
		if (gemv_host_trace_enabled(&host_trace))
			gemv_host_trace_measurement_begin(&operation_measurement);
		DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_rows_per_dpu * n_size_pad * sizeof(T) + n_size_pad * sizeof(T), max_rows_per_dpu * sizeof(T), DPU_XFER_DEFAULT));
		if (gemv_host_trace_enabled(&host_trace)) {
			gemv_host_trace_measurement_end(&operation_measurement);
			gemv_host_trace_record_transfer(
				&host_trace, "output_vector", "FROM_DPU", rep, warmup,
				"MRAM", "DPU_MRAM_HEAP_POINTER_NAME",
				(uint64_t)max_rows_per_dpu * n_size_pad * sizeof(T)
					+ (uint64_t)n_size_pad * sizeof(T),
				result_logical_bytes, 0,
				(uint64_t)max_rows_per_dpu * sizeof(T),
				rep, rep, "NONE", 0, &operation_measurement
			);
		}
		if(rep >= p.n_warmup)
			stop(&timer, 3);
	}
#if ENERGY
	double acc_energy, avg_energy, acc_time, avg_time;
	DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_ACCUMULATE, &acc_energy));
	DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_AVERAGE, &avg_energy));
	DPU_ASSERT(dpu_probe_get(&probe, DPU_TIME, DPU_ACCUMULATE, &acc_time));
	DPU_ASSERT(dpu_probe_get(&probe, DPU_TIME, DPU_AVERAGE, &avg_time));
#endif

	// Print timing results
	printf("CPU Version Time (ms): ");
	print(&timer, 0, 1);
	printf("CPU-DPU Time (ms): ");
	print(&timer, 1, p.n_reps);
	printf("DPU Kernel Time (ms): ");
	print(&timer, 2, p.n_reps);
	printf("DPU-CPU Time (ms): ");
	print(&timer, 3, p.n_reps);

#if ENERGY
	printf("Energy (J): %f J\t", avg_energy);
#endif

	// Check output
	bool status = true;
	unsigned int n,j;
	i = 0;
	for (n = 0; n < nr_of_dpus; n++) {
		for (j = 0; j < dpu_info[n].rows_per_dpu; j++) {
			if(C[i] != C_dpu[n * max_rows_per_dpu + j]) {
				status = false;
#if PRINT
	//			printf("%d: %d -- %d\n", i, C[i], C_dpu[n * max_rows_per_dpu + j]);
#endif
			}
			i++;
		}
	}
	if (status) {
		printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
	} else {
		printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
	}

	// Deallocation
	free(A);
	free(B);
	free(C);
	free(C_dpu);
	free(matrix_logical_bytes);
	free(result_logical_bytes);
	free(dpu_info);
	free(input_args);
	struct GemvHostTraceMeasurement free_measurement = {0};
	if (gemv_host_trace_enabled(&host_trace))
		gemv_host_trace_measurement_begin(&free_measurement);
	DPU_ASSERT(free_gemv_dpus(&dpu_allocation, dpu_set));
	if (gemv_host_trace_enabled(&host_trace)) {
		gemv_host_trace_measurement_end(&free_measurement);
		gemv_host_trace_record_event(
			&host_trace, "dpu_free", "", -1, -1,
			&free_measurement
		);
	}
	if (!gemv_host_trace_write(&host_trace)) {
		gemv_host_trace_destroy(&host_trace);
		return EXIT_FAILURE;
	}
	gemv_host_trace_destroy(&host_trace);

#if ENERGY
	DPU_ASSERT(dpu_probe_deinit(&probe));
#endif

	return status ? 0 : -1;
}
