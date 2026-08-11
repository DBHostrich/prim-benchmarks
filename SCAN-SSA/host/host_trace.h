#ifndef _SCAN_HOST_TRACE_H_
#define _SCAN_HOST_TRACE_H_

#include <dpu.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct ScanHostTraceMeasurement {
	uint64_t wall_start_ns;
	uint64_t wall_end_ns;
	uint64_t thread_cpu_start_ns;
	uint64_t thread_cpu_end_ns;
	int32_t cpu_id_start;
	int32_t cpu_id_end;
	uint64_t voluntary_context_switches_start;
	uint64_t voluntary_context_switches_end;
	uint64_t involuntary_context_switches_start;
	uint64_t involuntary_context_switches_end;
	uint64_t minor_faults_start;
	uint64_t minor_faults_end;
	uint64_t major_faults_start;
	uint64_t major_faults_end;
};

struct ScanHostTraceEvent {
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
	const char *diagnostic_copy_ordinal;
	uint64_t mram_push_ordinal_since_launch;
	uint64_t replay_delay_requested_us;
	uint64_t start_ns;
	uint64_t end_ns;
	uint64_t thread_cpu_ns;
	uint64_t wall_minus_thread_cpu_ns;
	int32_t cpu_id_start;
	int32_t cpu_id_end;
	uint64_t voluntary_context_switch_delta;
	uint64_t involuntary_context_switch_delta;
	uint64_t minor_fault_delta;
	uint64_t major_fault_delta;
};

struct ScanHostTraceDpu {
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

struct ScanHostTrace {
	bool enabled;
	const char *events_output_path;
	const char *dpus_output_path;
	const char *run_id;
	uint64_t repeat_id;
	uint32_t configured_dpus;
	uint32_t actual_ranks;
	uint32_t num_tasklets;
	uint64_t input_size;
	uint32_t input_size_per_dpu;
	uint32_t input_size_per_dpu_round;
	const char *scaling_mode;
	const char *host_numa_node;
	const char *process_state;
	const char *host_binding_mode;
	const char *host_cpu_list;
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
	struct ScanHostTraceEvent *events;
	size_t num_events;
	size_t event_capacity;
	struct ScanHostTraceDpu *dpu_rows;
	size_t num_dpu_rows;
	size_t dpu_row_capacity;
};

bool scan_host_trace_requested(void);
bool scan_host_trace_enabled(const struct ScanHostTrace *trace);
uint64_t scan_host_trace_now_ns(void);
void scan_host_trace_measurement_begin(struct ScanHostTraceMeasurement *measurement);
void scan_host_trace_measurement_end(struct ScanHostTraceMeasurement *measurement);

bool scan_host_trace_init(
	struct ScanHostTrace *trace,
	struct dpu_set_t dpu_set,
	uint32_t configured_dpus,
	uint32_t num_tasklets,
	uint64_t input_size,
	uint32_t input_size_per_dpu,
	uint32_t input_size_per_dpu_round,
	const char *scaling_mode
);

void scan_host_trace_record_event(
	struct ScanHostTrace *trace,
	const char *op,
	const char *subop,
	int32_t iteration,
	int32_t warmup,
	const struct ScanHostTraceMeasurement *measurement
);

void scan_host_trace_record_transfer(
	struct ScanHostTrace *trace,
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
	const char *diagnostic_copy_ordinal,
	uint64_t mram_push_ordinal_since_launch,
	uint64_t replay_delay_requested_us,
	const struct ScanHostTraceMeasurement *measurement
);

bool scan_host_trace_write(const struct ScanHostTrace *trace);
void scan_host_trace_destroy(struct ScanHostTrace *trace);

#endif
