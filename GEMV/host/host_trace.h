#ifndef _GEMV_HOST_TRACE_H_
#define _GEMV_HOST_TRACE_H_

#include <dpu.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct GemvHostTraceEvent {
	uint64_t event_id;
	const char *op;
	const char *subop;
	const char *direction;
	int32_t iteration;
	int32_t warmup;
	bool has_transfer;
	uint64_t size_per_dpu_bytes;
	uint64_t total_logical_bytes;
	uint64_t total_transfer_bytes;
	const char *target_space;
	const char *target_symbol;
	uint64_t offset_bytes;
	uint64_t source_buffer_use_count_before;
	uint64_t target_region_access_count_before;
	uint64_t start_ns;
	uint64_t end_ns;
};

struct GemvHostTraceDpu {
	uint64_t event_id;
	int32_t iteration;
	int32_t warmup;
	const char *op;
	const char *subop;
	const char *direction;
	uint32_t global_dpu_id;
	uint64_t logical_bytes;
	uint64_t transfer_bytes;
};

struct GemvHostTrace {
	bool enabled;
	const char *events_output_path;
	const char *dpus_output_path;
	const char *run_id;
	uint64_t repeat_id;
	uint32_t configured_dpus;
	uint32_t actual_ranks;
	uint32_t num_tasklets;
	uint32_t m_size;
	uint32_t n_size;
	uint32_t n_size_pad;
	uint32_t max_rows_per_dpu;
	const char *host_numa_node;
	const char *process_state;
	uint64_t pretrace_warmup_runs;
	uint32_t *rank_ordinals;
	uint32_t *dpu_ids_in_rank;
	uint32_t *sdk_physical_rank_ids;
	uint32_t *dpu_sysfs_rank_ids;
	uint32_t *dpu_rank_numa_nodes;
	uint32_t *dpu_channel_ids;
	uint32_t *dpu_ci_ids;
	uint32_t *dpu_member_ids;
	char *active_dpus_per_rank;
	char *dpu_rank_numa_nodes_text;
	char *dpu_channel_ids_text;
	char *dpu_sysfs_rank_ids_text;
	char *sdk_physical_rank_ids_text;
	char *allocated_topology_signature;
	char cpu_dpu_numa_relation[16];
	struct GemvHostTraceEvent *events;
	size_t num_events;
	size_t event_capacity;
	struct GemvHostTraceDpu *dpu_rows;
	size_t num_dpu_rows;
	size_t dpu_row_capacity;
};

bool gemv_host_trace_requested(void);
bool gemv_host_trace_enabled(const struct GemvHostTrace *trace);
uint64_t gemv_host_trace_now_ns(void);

bool gemv_host_trace_init(
	struct GemvHostTrace *trace,
	struct dpu_set_t dpu_set,
	uint32_t configured_dpus,
	uint32_t num_tasklets,
	uint32_t m_size,
	uint32_t n_size,
	uint32_t n_size_pad,
	uint32_t max_rows_per_dpu
);

void gemv_host_trace_record_event(
	struct GemvHostTrace *trace,
	const char *op,
	const char *subop,
	int32_t iteration,
	int32_t warmup,
	uint64_t start_ns,
	uint64_t end_ns
);

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
	uint64_t source_buffer_use_count_before,
	uint64_t target_region_access_count_before,
	uint64_t start_ns,
	uint64_t end_ns
);

bool gemv_host_trace_write(const struct GemvHostTrace *trace);
void gemv_host_trace_destroy(struct GemvHostTrace *trace);

#endif
