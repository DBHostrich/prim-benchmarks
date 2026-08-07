#define _GNU_SOURCE

#include "host_trace.h"

#include <dpu_management.h>

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct GemvPhysicalRankTopology {
	uint32_t sysfs_rank_id;
	uint32_t sdk_rank_id;
	uint32_t numa_node;
	uint32_t channel_id;
};

static bool parse_unsigned_env(
	const char *name,
	const char *value,
	uint64_t *parsed_value
) {
	char *end = NULL;
	unsigned long long parsed;

	if (value == NULL || value[0] == '\0') {
		*parsed_value = 0;
		return true;
	}
	errno = 0;
	parsed = strtoull(value, &end, 10);
	if (errno != 0 || end == value || *end != '\0') {
		fprintf(stderr, "Invalid %s: %s\n", name, value);
		return false;
	}
	*parsed_value = (uint64_t)parsed;
	return true;
}

static bool is_label_atom(const char *value) {
	const unsigned char *cursor = (const unsigned char *)value;
	if (value == NULL || value[0] == '\0')
		return false;
	for (; *cursor != '\0'; ++cursor) {
		if (!isalnum(*cursor) && *cursor != '_' && *cursor != '-'
			&& *cursor != '.')
			return false;
	}
	return true;
}

static bool lookup_physical_rank_topology(
	const char *path,
	uint32_t sdk_rank_id,
	struct GemvPhysicalRankTopology *topology
) {
	FILE *stream = fopen(path, "r");
	char line[1024];
	uint32_t matches = 0;

	if (stream == NULL) {
		fprintf(stderr, "Could not open DPU rank topology %s: %s\n",
			path, strerror(errno));
		return false;
	}
	while (fgets(line, sizeof(line), stream) != NULL) {
		char rank_path[256];
		char status[32];
		uint32_t sysfs_rank_id;
		uint32_t row_sdk_rank_id;
		uint32_t numa_node;
		uint32_t channel_id;
		uint32_t ci_count;
		uint32_t dpus_per_ci;
		uint32_t dpus_per_rank;
		uint64_t mram_per_dpu;
		uint64_t mram_per_rank;
		int parsed;

		if (strncmp(line, "rank_path", strlen("rank_path")) == 0)
			continue;
		parsed = sscanf(
			line,
			"%255s %" SCNu32 " %" SCNu32 " %" SCNu32 " %" SCNu32
			" %" SCNu32 " %" SCNu32 " %" SCNu32 " %" SCNu64
			" %" SCNu64 " %31s",
			rank_path, &sysfs_rank_id, &row_sdk_rank_id, &numa_node,
			&channel_id, &ci_count, &dpus_per_ci, &dpus_per_rank,
			&mram_per_dpu, &mram_per_rank, status
		);
		if (parsed == EOF || parsed == 0)
			continue;
		if (parsed != 11) {
			fprintf(stderr, "Malformed DPU rank topology row: %s", line);
			fclose(stream);
			return false;
		}
		if (row_sdk_rank_id != sdk_rank_id)
			continue;
		if (strcmp(status, "ok") != 0 || ci_count != 8 || dpus_per_ci != 8
			|| dpus_per_rank != 64) {
			fprintf(stderr,
				"Unsupported topology for SDK rank %" PRIu32
				": status=%s ci=%" PRIu32 " dpus_per_ci=%" PRIu32
				" dpus_per_rank=%" PRIu32 "\n",
				sdk_rank_id, status, ci_count, dpus_per_ci, dpus_per_rank);
			fclose(stream);
			return false;
		}
		topology->sysfs_rank_id = sysfs_rank_id;
		topology->sdk_rank_id = row_sdk_rank_id;
		topology->numa_node = numa_node;
		topology->channel_id = channel_id;
		++matches;
	}
	if (ferror(stream)) {
		fprintf(stderr, "Could not read DPU rank topology %s: %s\n",
			path, strerror(errno));
		fclose(stream);
		return false;
	}
	fclose(stream);
	if (matches != 1) {
		fprintf(stderr,
			"DPU rank topology has %" PRIu32
			" rows for allocated SDK rank %" PRIu32 "\n",
			matches, sdk_rank_id);
		return false;
	}
	return true;
}

static bool append_text(
	char *buffer,
	size_t capacity,
	size_t *offset,
	const char *format,
	...
) {
	va_list arguments;
	int written;

	if (*offset >= capacity)
		return false;
	va_start(arguments, format);
	written = vsnprintf(buffer + *offset, capacity - *offset, format, arguments);
	va_end(arguments);
	if (written < 0 || (size_t)written >= capacity - *offset)
		return false;
	*offset += (size_t)written;
	return true;
}

static const char *sdk_api_kind(const struct GemvHostTraceEvent *event) {
	return event->has_transfer ? "PUSH_XFER" : "";
}

static const char *logical_distribution_class(
	const struct GemvHostTraceEvent *event
) {
	if (!event->has_transfer)
		return "";
	if (strcmp(event->subop, "input_vector") == 0)
		return "SHARED_REPLICATION";
	if (strcmp(event->subop, "input_arguments") == 0
		|| strcmp(event->subop, "input_matrix") == 0)
		return "PARTITIONED_SCATTER";
	if (strcmp(event->subop, "output_vector") == 0)
		return "PARTITIONED_GATHER";
	return "UNKNOWN";
}

static const char *same_source_across_group(
	const struct GemvHostTraceEvent *event
) {
	if (!event->has_transfer)
		return "";
	return strcmp(event->subop, "input_vector") == 0 ? "1" : "0";
}

static const char *phase_class(const struct GemvHostTraceEvent *event) {
	if (!event->has_transfer)
		return "";
	return event->warmup == 1 ? "WARMUP" : "ITERATIVE";
}

static void write_csv_string(FILE *stream, const char *value) {
	const char *cursor;
	bool quote = false;

	if (value == NULL)
		return;
	for (cursor = value; *cursor != '\0'; ++cursor) {
		if (*cursor == ',' || *cursor == '"' || *cursor == '\n'
			|| *cursor == '\r') {
			quote = true;
			break;
		}
	}
	if (!quote) {
		fputs(value, stream);
		return;
	}
	fputc('"', stream);
	for (cursor = value; *cursor != '\0'; ++cursor) {
		if (*cursor == '"')
			fputc('"', stream);
		fputc(*cursor, stream);
	}
	fputc('"', stream);
}

static void grow_events(struct GemvHostTrace *trace) {
	size_t new_capacity = trace->event_capacity == 0
		? 64u : trace->event_capacity * 2u;
	struct GemvHostTraceEvent *events;

	if (new_capacity < trace->event_capacity
		|| new_capacity > SIZE_MAX / sizeof(*trace->events)) {
		fprintf(stderr, "GEMV host event trace capacity overflow\n");
		exit(EXIT_FAILURE);
	}
	events = realloc(trace->events, new_capacity * sizeof(*trace->events));
	if (events == NULL) {
		fprintf(stderr, "Could not grow the GEMV event trace\n");
		exit(EXIT_FAILURE);
	}
	memset(events + trace->event_capacity, 0,
		(new_capacity - trace->event_capacity) * sizeof(*events));
	trace->events = events;
	trace->event_capacity = new_capacity;
}

static void grow_dpu_rows(struct GemvHostTrace *trace) {
	size_t new_capacity = trace->dpu_row_capacity == 0
		? (size_t)trace->configured_dpus : trace->dpu_row_capacity * 2u;
	struct GemvHostTraceDpu *rows;

	if (new_capacity < trace->dpu_row_capacity
		|| new_capacity > SIZE_MAX / sizeof(*trace->dpu_rows)) {
		fprintf(stderr, "GEMV DPU trace capacity overflow\n");
		exit(EXIT_FAILURE);
	}
	rows = realloc(trace->dpu_rows, new_capacity * sizeof(*trace->dpu_rows));
	if (rows == NULL) {
		fprintf(stderr, "Could not grow the GEMV DPU trace\n");
		exit(EXIT_FAILURE);
	}
	memset(rows + trace->dpu_row_capacity, 0,
		(new_capacity - trace->dpu_row_capacity) * sizeof(*rows));
	trace->dpu_rows = rows;
	trace->dpu_row_capacity = new_capacity;
}

bool gemv_host_trace_requested(void) {
	const char *path = getenv("GEMV_TRACE_CSV");
	return path != NULL && path[0] != '\0';
}

bool gemv_host_trace_enabled(const struct GemvHostTrace *trace) {
	return trace != NULL && trace->enabled;
}

uint64_t gemv_host_trace_now_ns(void) {
	struct timespec now;
	if (clock_gettime(CLOCK_MONOTONIC_RAW, &now) != 0) {
		perror("clock_gettime(CLOCK_MONOTONIC_RAW)");
		exit(EXIT_FAILURE);
	}
	return (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
}

bool gemv_host_trace_init(
	struct GemvHostTrace *trace,
	struct dpu_set_t dpu_set,
	uint32_t configured_dpus,
	uint32_t num_tasklets,
	uint32_t m_size,
	uint32_t n_size,
	uint32_t n_size_pad,
	uint32_t max_rows_per_dpu
) {
	const char *events_path = getenv("GEMV_TRACE_CSV");
	const char *dpus_path = getenv("GEMV_TRACE_DPUS_CSV");
	const char *run_id = getenv("GEMV_TRACE_RUN_ID");
	const char *repeat_id = getenv("GEMV_TRACE_REPEAT_ID");
	const char *host_numa_node = getenv("GEMV_TRACE_HOST_NUMA_NODE");
	const char *process_state = getenv("GEMV_TRACE_PROCESS_STATE");
	const char *pretrace_warmup_runs = getenv("GEMV_TRACE_PREWARM_RUNS");
	const char *topology_path = getenv("GEMV_TRACE_DPU_RANK_TOPOLOGY_TSV");
	struct dpu_set_t rank = {0};
	struct dpu_set_t dpu = {0};
	struct dpu_rank_t *sdk_rank;
	struct GemvPhysicalRankTopology physical = {0};
	uint32_t rank_ordinal;
	uint32_t dpu_id_in_rank;
	uint32_t global_dpu_id = 0;
	uint32_t *rank_counts = NULL;
	size_t rank_text_capacity;
	size_t shape_offset = 0;
	size_t numa_offset = 0;
	size_t channel_offset = 0;
	size_t sysfs_offset = 0;
	size_t sdk_offset = 0;
	size_t signature_offset = 0;
	bool all_local = true;
	bool all_remote = true;
	char *host_end = NULL;
	unsigned long host_numa = 0;
	bool host_numa_valid = false;

	memset(trace, 0, sizeof(*trace));
	if (!gemv_host_trace_requested())
		return true;
	if (dpus_path == NULL || dpus_path[0] == '\0') {
		fprintf(stderr, "GEMV_TRACE_DPUS_CSV is required with GEMV_TRACE_CSV\n");
		return false;
	}
	if (topology_path == NULL || topology_path[0] == '\0') {
		fprintf(stderr,
			"GEMV_TRACE_DPU_RANK_TOPOLOGY_TSV is required for v8 labels\n");
		return false;
	}

	trace->events_output_path = events_path;
	trace->dpus_output_path = dpus_path;
	trace->run_id = run_id == NULL || run_id[0] == '\0' ? "gemv" : run_id;
	trace->configured_dpus = configured_dpus;
	trace->num_tasklets = num_tasklets;
	trace->m_size = m_size;
	trace->n_size = n_size;
	trace->n_size_pad = n_size_pad;
	trace->max_rows_per_dpu = max_rows_per_dpu;
	trace->host_numa_node = host_numa_node == NULL || host_numa_node[0] == '\0'
		? "unknown" : host_numa_node;
	trace->process_state = process_state == NULL || process_state[0] == '\0'
		? "fresh_process" : process_state;
	trace->event_capacity = 64u;
	trace->dpu_row_capacity = (size_t)configured_dpus * 16u;
	if (!parse_unsigned_env("GEMV_TRACE_REPEAT_ID", repeat_id, &trace->repeat_id)
		|| !parse_unsigned_env("GEMV_TRACE_PREWARM_RUNS", pretrace_warmup_runs,
			&trace->pretrace_warmup_runs))
		return false;
	if (!is_label_atom(trace->host_numa_node)
		|| !is_label_atom(trace->process_state)) {
		fprintf(stderr, "GEMV trace NUMA/process label contains unsupported characters\n");
		return false;
	}
	if (dpu_get_nr_ranks(dpu_set, &trace->actual_ranks) != DPU_OK
		|| trace->actual_ranks == 0 || configured_dpus == 0) {
		fprintf(stderr, "Could not query a valid GEMV allocation shape\n");
		return false;
	}

	rank_text_capacity = (size_t)trace->actual_ranks * 96u + 1u;
	trace->rank_ordinals = calloc(configured_dpus, sizeof(uint32_t));
	trace->dpu_ids_in_rank = calloc(configured_dpus, sizeof(uint32_t));
	trace->sdk_physical_rank_ids = calloc(configured_dpus, sizeof(uint32_t));
	trace->dpu_sysfs_rank_ids = calloc(configured_dpus, sizeof(uint32_t));
	trace->dpu_rank_numa_nodes = calloc(configured_dpus, sizeof(uint32_t));
	trace->dpu_channel_ids = calloc(configured_dpus, sizeof(uint32_t));
	trace->dpu_ci_ids = calloc(configured_dpus, sizeof(uint32_t));
	trace->dpu_member_ids = calloc(configured_dpus, sizeof(uint32_t));
	trace->active_dpus_per_rank = calloc(rank_text_capacity, 1);
	trace->dpu_rank_numa_nodes_text = calloc(rank_text_capacity, 1);
	trace->dpu_channel_ids_text = calloc(rank_text_capacity, 1);
	trace->dpu_sysfs_rank_ids_text = calloc(rank_text_capacity, 1);
	trace->sdk_physical_rank_ids_text = calloc(rank_text_capacity, 1);
	trace->allocated_topology_signature = calloc(rank_text_capacity, 1);
	trace->events = calloc(trace->event_capacity, sizeof(*trace->events));
	trace->dpu_rows = calloc(trace->dpu_row_capacity, sizeof(*trace->dpu_rows));
	rank_counts = calloc(trace->actual_ranks, sizeof(*rank_counts));
	if (trace->rank_ordinals == NULL || trace->dpu_ids_in_rank == NULL
		|| trace->sdk_physical_rank_ids == NULL
		|| trace->dpu_sysfs_rank_ids == NULL
		|| trace->dpu_rank_numa_nodes == NULL || trace->dpu_channel_ids == NULL
		|| trace->dpu_ci_ids == NULL || trace->dpu_member_ids == NULL
		|| trace->active_dpus_per_rank == NULL
		|| trace->dpu_rank_numa_nodes_text == NULL
		|| trace->dpu_channel_ids_text == NULL
		|| trace->dpu_sysfs_rank_ids_text == NULL
		|| trace->sdk_physical_rank_ids_text == NULL
		|| trace->allocated_topology_signature == NULL
		|| trace->events == NULL || trace->dpu_rows == NULL
		|| rank_counts == NULL) {
		fprintf(stderr, "Could not allocate GEMV trace buffers\n");
		free(rank_counts);
		gemv_host_trace_destroy(trace);
		return false;
	}

	if (strcmp(trace->host_numa_node, "unbound") != 0) {
		errno = 0;
		host_numa = strtoul(trace->host_numa_node, &host_end, 10);
		host_numa_valid = errno == 0 && host_end != trace->host_numa_node
			&& *host_end == '\0';
	}

	DPU_RANK_FOREACH(dpu_set, rank, rank_ordinal) {
		const char *separator = rank_ordinal == 0 ? "" : "|";
		if (rank_ordinal >= trace->actual_ranks) {
			fprintf(stderr, "Allocated GEMV rank ordinal exceeds rank count\n");
			free(rank_counts);
			gemv_host_trace_destroy(trace);
			return false;
		}
		sdk_rank = dpu_rank_from_set(rank);
		if (sdk_rank == NULL || !lookup_physical_rank_topology(
				topology_path, dpu_get_rank_id(sdk_rank), &physical)) {
			free(rank_counts);
			gemv_host_trace_destroy(trace);
			return false;
		}
		if (!append_text(trace->dpu_rank_numa_nodes_text, rank_text_capacity,
				&numa_offset, "%s%u", separator, physical.numa_node)
			|| !append_text(trace->dpu_channel_ids_text, rank_text_capacity,
				&channel_offset, "%s%u", separator, physical.channel_id)
			|| !append_text(trace->dpu_sysfs_rank_ids_text, rank_text_capacity,
				&sysfs_offset, "%s%u", separator, physical.sysfs_rank_id)
			|| !append_text(trace->sdk_physical_rank_ids_text, rank_text_capacity,
				&sdk_offset, "%s%u", separator, physical.sdk_rank_id)
			|| !append_text(trace->allocated_topology_signature,
				rank_text_capacity, &signature_offset,
				"%sr%u@n%u@c%u", separator, physical.sysfs_rank_id,
				physical.numa_node, physical.channel_id)) {
			fprintf(stderr, "Could not format GEMV topology signature\n");
			free(rank_counts);
			gemv_host_trace_destroy(trace);
			return false;
		}
		if (host_numa_valid) {
			all_local = all_local && host_numa == physical.numa_node;
			all_remote = all_remote && host_numa != physical.numa_node;
		}
		DPU_FOREACH(rank, dpu, dpu_id_in_rank) {
			if (global_dpu_id >= configured_dpus) {
				fprintf(stderr, "Allocated GEMV topology exceeds configured DPUs\n");
				free(rank_counts);
				gemv_host_trace_destroy(trace);
				return false;
			}
			trace->rank_ordinals[global_dpu_id] = rank_ordinal;
			trace->dpu_ids_in_rank[global_dpu_id] = dpu_id_in_rank;
			trace->sdk_physical_rank_ids[global_dpu_id] = physical.sdk_rank_id;
			trace->dpu_sysfs_rank_ids[global_dpu_id] = physical.sysfs_rank_id;
			trace->dpu_rank_numa_nodes[global_dpu_id] = physical.numa_node;
			trace->dpu_channel_ids[global_dpu_id] = physical.channel_id;
			trace->dpu_ci_ids[global_dpu_id] = dpu_get_slice_id(dpu.dpu);
			trace->dpu_member_ids[global_dpu_id] = dpu_get_member_id(dpu.dpu);
			++rank_counts[rank_ordinal];
			++global_dpu_id;
		}
	}
	if (global_dpu_id != configured_dpus) {
		fprintf(stderr, "Allocated GEMV topology has %u DPUs, expected %u\n",
			global_dpu_id, configured_dpus);
		free(rank_counts);
		gemv_host_trace_destroy(trace);
		return false;
	}
	for (rank_ordinal = 0; rank_ordinal < trace->actual_ranks; ++rank_ordinal) {
		if (!append_text(trace->active_dpus_per_rank, rank_text_capacity,
				&shape_offset, "%s%u", rank_ordinal == 0 ? "" : "|",
				rank_counts[rank_ordinal])) {
			fprintf(stderr, "Could not format GEMV rank shape\n");
			free(rank_counts);
			gemv_host_trace_destroy(trace);
			return false;
		}
	}
	free(rank_counts);

	if (strcmp(trace->host_numa_node, "unbound") == 0)
		strcpy(trace->cpu_dpu_numa_relation, "UNBOUND");
	else if (!host_numa_valid)
		strcpy(trace->cpu_dpu_numa_relation, "UNKNOWN");
	else if (all_local)
		strcpy(trace->cpu_dpu_numa_relation, "LOCAL");
	else if (all_remote)
		strcpy(trace->cpu_dpu_numa_relation, "REMOTE");
	else
		strcpy(trace->cpu_dpu_numa_relation, "MIXED");

	trace->enabled = true;
	return true;
}

void gemv_host_trace_record_event(
	struct GemvHostTrace *trace,
	const char *op,
	const char *subop,
	int32_t iteration,
	int32_t warmup,
	uint64_t start_ns,
	uint64_t end_ns
) {
	struct GemvHostTraceEvent *event;
	if (!gemv_host_trace_enabled(trace))
		return;
	if (end_ns < start_ns) {
		fprintf(stderr, "GEMV host trace clock moved backwards\n");
		exit(EXIT_FAILURE);
	}
	if (trace->num_events >= trace->event_capacity)
		grow_events(trace);
	event = &trace->events[trace->num_events];
	event->event_id = trace->num_events;
	event->op = op;
	event->subop = subop;
	event->iteration = iteration;
	event->warmup = warmup;
	event->start_ns = start_ns;
	event->end_ns = end_ns;
	++trace->num_events;
}

void gemv_host_trace_record_transfer(
	struct GemvHostTrace *trace,
	const char *subop,
	const char *direction,
	int32_t iteration,
	int32_t warmup,
	const char *target_space,
	const char *target_symbol,
	uint64_t offset_bytes,
	const uint64_t *logical_bytes_per_dpu,
	uint64_t uniform_logical_bytes,
	uint64_t size_per_dpu_bytes,
	uint64_t start_ns,
	uint64_t end_ns
) {
	struct GemvHostTraceEvent *event;
	uint64_t event_id;
	uint64_t total_logical_bytes = 0;
	uint32_t dpu_id;

	if (!gemv_host_trace_enabled(trace))
		return;
	if (end_ns < start_ns
		|| size_per_dpu_bytes > UINT64_MAX / trace->configured_dpus) {
		fprintf(stderr, "Invalid GEMV transfer trace interval or byte count\n");
		exit(EXIT_FAILURE);
	}
	for (dpu_id = 0; dpu_id < trace->configured_dpus; ++dpu_id) {
		uint64_t logical_bytes = logical_bytes_per_dpu == NULL
			? uniform_logical_bytes : logical_bytes_per_dpu[dpu_id];
		if (logical_bytes > size_per_dpu_bytes
			|| total_logical_bytes > UINT64_MAX - logical_bytes) {
			fprintf(stderr, "Invalid GEMV logical bytes for DPU %u\n", dpu_id);
			exit(EXIT_FAILURE);
		}
		total_logical_bytes += logical_bytes;
	}
	if (trace->num_events >= trace->event_capacity)
		grow_events(trace);
	event_id = trace->num_events;
	event = &trace->events[event_id];
	event->event_id = event_id;
	event->op = "dpu_push_xfer";
	event->subop = subop;
	event->direction = direction;
	event->iteration = iteration;
	event->warmup = warmup;
	event->has_transfer = true;
	event->size_per_dpu_bytes = size_per_dpu_bytes;
	event->total_logical_bytes = total_logical_bytes;
	event->total_transfer_bytes = size_per_dpu_bytes * trace->configured_dpus;
	event->target_space = target_space;
	event->target_symbol = target_symbol;
	event->offset_bytes = offset_bytes;
	event->start_ns = start_ns;
	event->end_ns = end_ns;
	++trace->num_events;

	for (dpu_id = 0; dpu_id < trace->configured_dpus; ++dpu_id) {
		struct GemvHostTraceDpu *row;
		if (trace->num_dpu_rows >= trace->dpu_row_capacity)
			grow_dpu_rows(trace);
		row = &trace->dpu_rows[trace->num_dpu_rows++];
		row->event_id = event_id;
		row->iteration = iteration;
		row->warmup = warmup;
		row->op = event->op;
		row->subop = subop;
		row->direction = direction;
		row->global_dpu_id = dpu_id;
		row->logical_bytes = logical_bytes_per_dpu == NULL
			? uniform_logical_bytes : logical_bytes_per_dpu[dpu_id];
		row->transfer_bytes = size_per_dpu_bytes;
	}
}

static bool format_transport_key(
	const struct GemvHostTrace *trace,
	const struct GemvHostTraceEvent *event,
	char *output,
	size_t output_size
) {
	int result;
	if (!event->has_transfer) {
		output[0] = '\0';
		return true;
	}
	result = snprintf(
		output, output_size,
		"v8;op=%s;direction=%s;sdk_api_kind=PUSH_XFER;timing_scope=PUSH_ONLY;"
		"logical_distribution_class=%s;target_space=%s;"
		"transfer_bytes_per_dpu=%" PRIu64 ";active_dpus=%u;"
		"active_ranks=%u;active_dpus_per_rank=%s;"
		"same_source_across_group=%s;host_numa_node=%s;"
		"dpu_rank_numa_nodes=%s;cpu_dpu_numa_relation=%s;"
		"dpu_channel_ids=%s;dpu_sysfs_rank_ids=%s;"
		"dpu_ci_ids=0-7;dpu_member_ids=0-7;"
		"allocated_topology_signature=%s;allocated_dpus=%u;"
		"allocated_ranks=%u;previous_sdk_mux_domain_class=COLLECTION",
		event->op, event->direction, logical_distribution_class(event),
		event->target_space, event->size_per_dpu_bytes,
		trace->configured_dpus, trace->actual_ranks,
		trace->active_dpus_per_rank, same_source_across_group(event),
		trace->host_numa_node, trace->dpu_rank_numa_nodes_text,
		trace->cpu_dpu_numa_relation, trace->dpu_channel_ids_text,
		trace->dpu_sysfs_rank_ids_text, trace->allocated_topology_signature,
		trace->configured_dpus, trace->actual_ranks
	);
	return result >= 0 && (size_t)result < output_size;
}

static bool write_events(const struct GemvHostTrace *trace) {
	FILE *stream = fopen(trace->events_output_path, "w");
	size_t index;
	if (stream == NULL) {
		fprintf(stderr, "Could not open GEMV event trace %s: %s\n",
			trace->events_output_path, strerror(errno));
		return false;
	}
	fputs(
		"run_id,repeat_id,event_id,configured_dpus,actual_ranks,num_tasklets,"
		"m_size,n_size,n_size_pad,max_rows_per_dpu,op,direction,sdk_api_kind,"
		"timing_scope,logical_distribution_class,target_space,"
		"transfer_bytes_per_dpu,active_dpus,active_ranks,active_dpus_per_rank,"
		"rank_ordinal,dpu_id_in_rank,same_source_across_group,host_numa_node,"
		"dpu_rank_numa_nodes,cpu_dpu_numa_relation,dpu_channel_ids,"
		"dpu_sysfs_rank_ids,sdk_physical_rank_ids,dpu_ci_ids,dpu_member_ids,"
		"allocated_topology_signature,allocated_dpus,allocated_ranks,"
		"previous_sdk_op,previous_sdk_direction,previous_sdk_transfer_bytes,"
		"previous_sdk_mux_domain_class,phase_class,subop,iteration,warmup,"
		"size_per_dpu_bytes,total_logical_bytes,total_transfer_bytes,"
		"target_symbol,offset_bytes,process_state,pretrace_warmup_runs,"
		"transport_key,host_start_ns,host_end_ns,measured_ns\n",
		stream
	);
	for (index = 0; index < trace->num_events; ++index) {
		const struct GemvHostTraceEvent *event = &trace->events[index];
		const struct GemvHostTraceEvent *previous = index == 0
			? NULL : &trace->events[index - 1];
		char transport_key[2048];

		if (!format_transport_key(trace, event, transport_key,
				sizeof(transport_key))) {
			fprintf(stderr, "GEMV v8 transport key is too long\n");
			fclose(stream);
			return false;
		}
		write_csv_string(stream, trace->run_id);
		fprintf(stream, ",%" PRIu64 ",%" PRIu64 ",%u,%u,%u,%u,%u,%u,%u,",
			trace->repeat_id, event->event_id, trace->configured_dpus,
			trace->actual_ranks, trace->num_tasklets, trace->m_size,
			trace->n_size, trace->n_size_pad, trace->max_rows_per_dpu);
		write_csv_string(stream, event->op);
		fputc(',', stream);
		write_csv_string(stream, event->direction);
		fputc(',', stream);
		write_csv_string(stream, sdk_api_kind(event));
		fputc(',', stream);
		if (event->has_transfer) {
			fputs("PUSH_ONLY,", stream);
			write_csv_string(stream, logical_distribution_class(event));
			fputc(',', stream);
			write_csv_string(stream, event->target_space);
			fprintf(stream, ",%" PRIu64 ",%u,%u,",
				event->size_per_dpu_bytes, trace->configured_dpus,
				trace->actual_ranks);
			write_csv_string(stream, trace->active_dpus_per_rank);
			fputs(",ALL,ALL,", stream);
			write_csv_string(stream, same_source_across_group(event));
			fputc(',', stream);
			write_csv_string(stream, trace->host_numa_node);
			fputc(',', stream);
			write_csv_string(stream, trace->dpu_rank_numa_nodes_text);
			fputc(',', stream);
			write_csv_string(stream, trace->cpu_dpu_numa_relation);
			fputc(',', stream);
			write_csv_string(stream, trace->dpu_channel_ids_text);
			fputc(',', stream);
			write_csv_string(stream, trace->dpu_sysfs_rank_ids_text);
			fputc(',', stream);
			write_csv_string(stream, trace->sdk_physical_rank_ids_text);
			fputs(",0-7,0-7,", stream);
			write_csv_string(stream, trace->allocated_topology_signature);
			fprintf(stream, ",%u,%u,", trace->configured_dpus,
				trace->actual_ranks);
			write_csv_string(stream, previous == NULL ? "NONE" : previous->op);
			fputc(',', stream);
			write_csv_string(stream,
				previous == NULL || previous->direction == NULL
					? "NONE" : previous->direction);
			fprintf(stream, ",%" PRIu64 ",COLLECTION,",
				previous != NULL && previous->has_transfer
					? previous->size_per_dpu_bytes : UINT64_C(0));
			write_csv_string(stream, phase_class(event));
		} else {
			fputs(",,,,,,,,,,", stream);
			write_csv_string(stream, trace->host_numa_node);
			fputs(",,,,,,,,,,,,,,,", stream);
		}
		fputc(',', stream);
		write_csv_string(stream, event->subop);
		fputc(',', stream);
		if (event->iteration >= 0)
			fprintf(stream, "%" PRId32, event->iteration);
		fputc(',', stream);
		if (event->warmup >= 0)
			fprintf(stream, "%" PRId32, event->warmup);
		fputc(',', stream);
		if (event->has_transfer) {
			fprintf(stream, "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",",
				event->size_per_dpu_bytes, event->total_logical_bytes,
				event->total_transfer_bytes);
			write_csv_string(stream, event->target_symbol);
			fprintf(stream, ",%" PRIu64, event->offset_bytes);
		} else {
			fputs(",,,,", stream);
		}
		fputc(',', stream);
		write_csv_string(stream, trace->process_state);
		fprintf(stream, ",%" PRIu64 ",", trace->pretrace_warmup_runs);
		write_csv_string(stream, transport_key);
		fprintf(stream, ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
			event->start_ns, event->end_ns, event->end_ns - event->start_ns);
	}
	if (fclose(stream) != 0) {
		fprintf(stderr, "Could not close GEMV event trace %s: %s\n",
			trace->events_output_path, strerror(errno));
		return false;
	}
	return true;
}

static bool write_dpu_rows(const struct GemvHostTrace *trace) {
	FILE *stream = fopen(trace->dpus_output_path, "w");
	size_t index;
	if (stream == NULL) {
		fprintf(stderr, "Could not open GEMV DPU trace %s: %s\n",
			trace->dpus_output_path, strerror(errno));
		return false;
	}
	fputs(
		"run_id,repeat_id,event_id,configured_dpus,actual_ranks,num_tasklets,"
		"iteration,warmup,op,subop,direction,global_dpu_id,rank_ordinal,"
		"dpu_id_in_rank,sdk_physical_rank_id,dpu_sysfs_rank_id,"
		"dpu_rank_numa_node,dpu_channel_id,dpu_ci_id,dpu_member_id,"
		"logical_bytes,transfer_bytes\n",
		stream
	);
	for (index = 0; index < trace->num_dpu_rows; ++index) {
		const struct GemvHostTraceDpu *row = &trace->dpu_rows[index];
		uint32_t dpu_id = row->global_dpu_id;
		write_csv_string(stream, trace->run_id);
		fprintf(stream,
			",%" PRIu64 ",%" PRIu64 ",%u,%u,%u,%" PRId32 ",%" PRId32 ",",
			trace->repeat_id, row->event_id, trace->configured_dpus,
			trace->actual_ranks, trace->num_tasklets, row->iteration,
			row->warmup);
		write_csv_string(stream, row->op);
		fputc(',', stream);
		write_csv_string(stream, row->subop);
		fputc(',', stream);
		write_csv_string(stream, row->direction);
		fprintf(stream, ",%u,%u,%u,%u,%u,%u,%u,%u,%u,%" PRIu64 ",%" PRIu64
			"\n",
			dpu_id, trace->rank_ordinals[dpu_id], trace->dpu_ids_in_rank[dpu_id],
			trace->sdk_physical_rank_ids[dpu_id], trace->dpu_sysfs_rank_ids[dpu_id],
			trace->dpu_rank_numa_nodes[dpu_id], trace->dpu_channel_ids[dpu_id],
			trace->dpu_ci_ids[dpu_id], trace->dpu_member_ids[dpu_id],
			row->logical_bytes, row->transfer_bytes);
	}
	if (fclose(stream) != 0) {
		fprintf(stderr, "Could not close GEMV DPU trace %s: %s\n",
			trace->dpus_output_path, strerror(errno));
		return false;
	}
	return true;
}

bool gemv_host_trace_write(const struct GemvHostTrace *trace) {
	if (!gemv_host_trace_enabled(trace))
		return true;
	return write_events(trace) && write_dpu_rows(trace);
}

void gemv_host_trace_destroy(struct GemvHostTrace *trace) {
	if (trace == NULL)
		return;
	free(trace->rank_ordinals);
	free(trace->dpu_ids_in_rank);
	free(trace->sdk_physical_rank_ids);
	free(trace->dpu_sysfs_rank_ids);
	free(trace->dpu_rank_numa_nodes);
	free(trace->dpu_channel_ids);
	free(trace->dpu_ci_ids);
	free(trace->dpu_member_ids);
	free(trace->active_dpus_per_rank);
	free(trace->dpu_rank_numa_nodes_text);
	free(trace->dpu_channel_ids_text);
	free(trace->dpu_sysfs_rank_ids_text);
	free(trace->sdk_physical_rank_ids_text);
	free(trace->allocated_topology_signature);
	free(trace->events);
	free(trace->dpu_rows);
	memset(trace, 0, sizeof(*trace));
}
