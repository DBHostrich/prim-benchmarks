#define _GNU_SOURCE

/**
* app.c
* VA Host Application Source File
*
*/
#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <sched.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#include <dpu.h>
#include <dpu_log.h>

#include "../support/common.h"
#include "../support/params.h"

#ifndef DPU_BINARY
#define DPU_BINARY "./bin/dpu_code"
#endif

#ifndef VA_VALIDATION_INPUT
#define VA_VALIDATION_INPUT 0
#endif

_Static_assert(sizeof(dpu_arguments_t) == 12, "VA argument payload must remain 12 bytes");

#if ENERGY
#include <dpu_probe.h>
#endif

static T *A;
static T *B;
static T *C;
static T *C2;

enum transfer_component_id {
    COMPONENT_PREPARE_ARGS,
    COMPONENT_PUSH_ARGS,
    COMPONENT_PREPARE_A,
    COMPONENT_PUSH_A,
    COMPONENT_PREPARE_B,
    COMPONENT_PUSH_B,
    COMPONENT_PREPARE_C,
    COMPONENT_PUSH_C,
    TRANSFER_COMPONENT_COUNT,
};

struct timing_snapshot {
    uint64_t wall_ns;
    uint64_t thread_cpu_ns;
    uint64_t process_cpu_ns;
    int cpu_id;
    long thread_nvcsw;
    long thread_nivcsw;
    long process_nvcsw;
    long process_nivcsw;
};

struct transfer_component_timing {
    struct timing_snapshot start;
    struct timing_snapshot end;
};

struct transfer_component_description {
    const char *component;
    const char *direction;
    const char *target_space;
    const char *operation;
};

static const struct transfer_component_description transfer_components[TRANSFER_COMPONENT_COUNT] = {
    [COMPONENT_PREPARE_ARGS] = { "PREPARE_ARGS", "H2D", "WRAM", "PREPARE" },
    [COMPONENT_PUSH_ARGS] = { "PUSH_ARGS", "H2D", "WRAM", "PUSH" },
    [COMPONENT_PREPARE_A] = { "PREPARE_A", "H2D", "MRAM", "PREPARE" },
    [COMPONENT_PUSH_A] = { "PUSH_A", "H2D", "MRAM", "PUSH" },
    [COMPONENT_PREPARE_B] = { "PREPARE_B", "H2D", "MRAM", "PREPARE" },
    [COMPONENT_PUSH_B] = { "PUSH_B", "H2D", "MRAM", "PUSH" },
    [COMPONENT_PREPARE_C] = { "PREPARE_C", "D2H", "MRAM", "PREPARE" },
    [COMPONENT_PUSH_C] = { "PUSH_C", "D2H", "MRAM", "PUSH" },
};

static uint64_t
raw_time_ns(void)
{
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC_RAW, &now) != 0) {
        perror("clock_gettime(CLOCK_MONOTONIC_RAW)");
        exit(EXIT_FAILURE);
    }
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
}

static uint64_t
clock_time_ns(clockid_t clock_id, const char *clock_name)
{
    struct timespec now;

    if (clock_gettime(clock_id, &now) != 0) {
        fprintf(stderr, "clock_gettime(%s): %s\n", clock_name, strerror(errno));
        exit(EXIT_FAILURE);
    }
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
}

static struct timing_snapshot
take_timing_snapshot(void)
{
    struct timing_snapshot snapshot;
    struct rusage thread_usage;
    struct rusage process_usage;

    snapshot.thread_cpu_ns = clock_time_ns(CLOCK_THREAD_CPUTIME_ID, "CLOCK_THREAD_CPUTIME_ID");
    snapshot.process_cpu_ns = clock_time_ns(CLOCK_PROCESS_CPUTIME_ID, "CLOCK_PROCESS_CPUTIME_ID");
    if (getrusage(RUSAGE_THREAD, &thread_usage) != 0) {
        perror("getrusage(RUSAGE_THREAD)");
        exit(EXIT_FAILURE);
    }
    if (getrusage(RUSAGE_SELF, &process_usage) != 0) {
        perror("getrusage(RUSAGE_SELF)");
        exit(EXIT_FAILURE);
    }
    snapshot.cpu_id = sched_getcpu();
    if (snapshot.cpu_id < 0) {
        perror("sched_getcpu");
        exit(EXIT_FAILURE);
    }
    snapshot.thread_nvcsw = thread_usage.ru_nvcsw;
    snapshot.thread_nivcsw = thread_usage.ru_nivcsw;
    snapshot.process_nvcsw = process_usage.ru_nvcsw;
    snapshot.process_nivcsw = process_usage.ru_nivcsw;
    snapshot.wall_ns = raw_time_ns();
    return snapshot;
}

static const char *
env_or(const char *name, const char *fallback)
{
    const char *value = getenv(name);
    return value != NULL && value[0] != '\0' ? value : fallback;
}

static void
write_csv_field(FILE *stream, const char *value)
{
    const char *cursor;

    fputc('"', stream);
    for (cursor = value; *cursor != '\0'; ++cursor) {
        if (*cursor == '"')
            fputc('"', stream);
        fputc(*cursor, stream);
    }
    fputc('"', stream);
}

static void
write_trace_row(
    FILE *stream,
    const char *run_id,
    uint32_t repeat_id,
    const char *configuration,
    uint32_t actual_dpus,
    uint32_t actual_ranks,
    unsigned int input_elements,
    const char *phase,
    uint64_t duration_ns,
    uint64_t bytes_per_dpu,
    uint64_t aggregate_bytes,
    const char *call_sequence,
    bool result_ok,
    uint64_t expected_checksum,
    uint64_t actual_checksum)
{
    fprintf(stream, "upmem.va_baseline.hw.v1,");
    write_csv_field(stream, run_id);
    fprintf(stream, ",%u,", repeat_id);
    write_csv_field(stream, configuration);
    fprintf(stream, ",%u,%u,%u,%u,%u,strong,", actual_dpus, actual_ranks,
        NR_TASKLETS, BL, input_elements);
    write_csv_field(stream, phase);
    fprintf(stream, ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",",
        duration_ns, bytes_per_dpu, aggregate_bytes);
    write_csv_field(stream, call_sequence);
    fprintf(stream, ",%u,%" PRIu64 ",%" PRIu64 ",",
        result_ok ? 1U : 0U, expected_checksum, actual_checksum);
    write_csv_field(stream, env_or("VA_HOST_NUMA_NODE", "unknown"));
    fputc(',', stream);
    write_csv_field(stream, env_or("VA_SDK_VERSION", "unknown"));
    fputc(',', stream);
    write_csv_field(stream, env_or("VA_SOURCE_SHA256", "unknown"));
    fputc(',', stream);
    write_csv_field(stream, env_or("VA_HOST_BINARY_SHA256", "unknown"));
    fputc(',', stream);
    write_csv_field(stream, env_or("VA_DPU_BINARY_SHA256", "unknown"));
    fputc('\n', stream);
}

static void
write_trace(
    const char *trace_path,
    const char *configuration,
    const char *transfer_order,
    uint32_t actual_dpus,
    uint32_t actual_ranks,
    unsigned int input_elements,
    unsigned int per_dpu_elements,
    int measured_repetitions,
    const uint64_t *cpu_ns,
    const uint64_t *h2d_ns,
    const uint64_t *kernel_ns,
    const uint64_t *d2h_ns,
    uint64_t setup_ns,
    uint64_t verify_ns,
    bool result_ok,
    uint64_t expected_checksum,
    uint64_t actual_checksum)
{
    FILE *stream;
    uint64_t payload_bytes = (uint64_t)per_dpu_elements * sizeof(T);
    uint64_t h2d_bytes_per_dpu = sizeof(dpu_arguments_t) + 2 * payload_bytes;
    uint64_t d2h_bytes_per_dpu = payload_bytes;
    uint64_t repeat_base = strtoull(env_or("VA_TRACE_REPEAT_ID", "1"), NULL, 10);
    const char *run_id = env_or("VA_TRACE_RUN_ID", configuration);
    const char *h2d_sequence = strcmp(transfer_order, "BA") == 0
        ? "prepare_args>push_args>prepare_B>push_B>prepare_A>push_A"
        : "prepare_args>push_args>prepare_A>push_A>prepare_B>push_B";

    stream = fopen(trace_path, "w");
    if (stream == NULL) {
        fprintf(stderr, "cannot open VA trace %s: %s\n", trace_path, strerror(errno));
        exit(EXIT_FAILURE);
    }
    fprintf(stream,
        "schema_version,run_id,repeat_id,configuration,actual_dpus,actual_ranks,"
        "tasklets,block_size_log2,input_elements,scaling,phase,duration_ns,"
        "bytes_per_dpu,aggregate_bytes,call_sequence,result_ok,expected_checksum,"
        "actual_checksum,host_numa_node,sdk_version,source_sha256,"
        "host_binary_sha256,dpu_binary_sha256\n");

    for (int rep = 0; rep < measured_repetitions; ++rep) {
        uint32_t repeat_id = (uint32_t)(repeat_base + (uint64_t)rep);
        uint64_t total_ns = h2d_ns[rep] + kernel_ns[rep] + d2h_ns[rep];

        write_trace_row(stream, run_id, repeat_id, configuration, actual_dpus,
            actual_ranks, input_elements, "SETUP", setup_ns, 0, 0,
            "dpu_alloc>dpu_load", result_ok, expected_checksum, actual_checksum);
        write_trace_row(stream, run_id, repeat_id, configuration, actual_dpus,
            actual_ranks, input_elements, "CPU_REFERENCE", cpu_ns[rep], 0, 0,
            "vector_addition_host", result_ok, expected_checksum, actual_checksum);
        write_trace_row(stream, run_id, repeat_id, configuration, actual_dpus,
            actual_ranks, input_elements, "H2D", h2d_ns[rep], h2d_bytes_per_dpu,
            h2d_bytes_per_dpu * actual_dpus, h2d_sequence,
            result_ok, expected_checksum, actual_checksum);
        write_trace_row(stream, run_id, repeat_id, configuration, actual_dpus,
            actual_ranks, input_elements, "KERNEL", kernel_ns[rep], 0, 0,
            "dpu_launch_synchronous", result_ok, expected_checksum, actual_checksum);
        write_trace_row(stream, run_id, repeat_id, configuration, actual_dpus,
            actual_ranks, input_elements, "D2H", d2h_ns[rep], d2h_bytes_per_dpu,
            d2h_bytes_per_dpu * actual_dpus, "prepare_C>push_C",
            result_ok, expected_checksum, actual_checksum);
        write_trace_row(stream, run_id, repeat_id, configuration, actual_dpus,
            actual_ranks, input_elements, "VERIFY", verify_ns, 0, 0,
            "elementwise_compare", result_ok, expected_checksum, actual_checksum);
        write_trace_row(stream, run_id, repeat_id, configuration, actual_dpus,
            actual_ranks, input_elements, "TOTAL", total_ns,
            h2d_bytes_per_dpu + d2h_bytes_per_dpu,
            (h2d_bytes_per_dpu + d2h_bytes_per_dpu) * actual_dpus,
            "H2D>KERNEL>D2H", result_ok, expected_checksum, actual_checksum);
    }
    if (fclose(stream) != 0) {
        perror("fclose VA trace");
        exit(EXIT_FAILURE);
    }
}

static uint64_t
snapshot_delta(uint64_t end, uint64_t start, const char *metric)
{
    if (end < start) {
        fprintf(stderr, "%s moved backwards\n", metric);
        exit(EXIT_FAILURE);
    }
    return end - start;
}

static long
counter_delta(long end, long start, const char *metric)
{
    if (end < start) {
        fprintf(stderr, "%s moved backwards\n", metric);
        exit(EXIT_FAILURE);
    }
    return end - start;
}

static void
write_transfer_breakdown(
    const char *trace_path,
    const char *configuration,
    uint32_t actual_dpus,
    uint32_t actual_ranks,
    unsigned int input_elements,
    unsigned int per_dpu_elements,
    int measured_repetitions,
    const struct transfer_component_timing *timings,
    bool result_ok)
{
    FILE *stream;
    uint64_t payload_bytes = (uint64_t)per_dpu_elements * sizeof(T);
    uint64_t repeat_base = strtoull(env_or("VA_TRACE_REPEAT_ID", "1"), NULL, 10);
    const char *run_id = env_or("VA_TRACE_RUN_ID", configuration);

    stream = fopen(trace_path, "w");
    if (stream == NULL) {
        fprintf(stderr, "cannot open VA transfer trace %s: %s\n", trace_path, strerror(errno));
        exit(EXIT_FAILURE);
    }
    fprintf(stream,
        "schema_version,run_id,repeat_id,configuration,actual_dpus,actual_ranks,"
        "tasklets,block_size_log2,input_elements,component,direction,target_space,"
        "operation,wall_start_ns,wall_end_ns,wall_duration_ns,thread_cpu_start_ns,"
        "thread_cpu_end_ns,thread_cpu_duration_ns,process_cpu_start_ns,"
        "process_cpu_end_ns,process_cpu_duration_ns,cpu_start,cpu_end,"
        "thread_nvcsw_delta,thread_nivcsw_delta,process_nvcsw_delta,"
        "process_nivcsw_delta,bytes_per_dpu,aggregate_bytes,result_ok,"
        "source_sha256,host_binary_sha256,dpu_binary_sha256\n");

    for (int rep = 0; rep < measured_repetitions; ++rep) {
        uint32_t repeat_id = (uint32_t)(repeat_base + (uint64_t)rep);
        for (int component = 0; component < TRANSFER_COMPONENT_COUNT; ++component) {
            const struct transfer_component_description *description =
                &transfer_components[component];
            const struct transfer_component_timing *timing =
                &timings[(size_t)rep * TRANSFER_COMPONENT_COUNT + component];
            uint64_t bytes_per_dpu = component <= COMPONENT_PUSH_ARGS
                ? sizeof(dpu_arguments_t)
                : payload_bytes;

            fprintf(stream, "upmem.va_transfer_breakdown.hw.v1,");
            write_csv_field(stream, run_id);
            fprintf(stream, ",%u,", repeat_id);
            write_csv_field(stream, configuration);
            fprintf(stream, ",%u,%u,%u,%u,%u,", actual_dpus, actual_ranks,
                NR_TASKLETS, BL, input_elements);
            write_csv_field(stream, description->component);
            fputc(',', stream);
            write_csv_field(stream, description->direction);
            fputc(',', stream);
            write_csv_field(stream, description->target_space);
            fputc(',', stream);
            write_csv_field(stream, description->operation);
            fprintf(stream,
                ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
                ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
                ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
                ",%d,%d,%ld,%ld,%ld,%ld,%" PRIu64 ",%" PRIu64 ",%u,",
                timing->start.wall_ns,
                timing->end.wall_ns,
                snapshot_delta(timing->end.wall_ns, timing->start.wall_ns, "wall clock"),
                timing->start.thread_cpu_ns,
                timing->end.thread_cpu_ns,
                snapshot_delta(timing->end.thread_cpu_ns, timing->start.thread_cpu_ns,
                    "thread CPU clock"),
                timing->start.process_cpu_ns,
                timing->end.process_cpu_ns,
                snapshot_delta(timing->end.process_cpu_ns, timing->start.process_cpu_ns,
                    "process CPU clock"),
                timing->start.cpu_id,
                timing->end.cpu_id,
                counter_delta(timing->end.thread_nvcsw, timing->start.thread_nvcsw,
                    "thread voluntary context switches"),
                counter_delta(timing->end.thread_nivcsw, timing->start.thread_nivcsw,
                    "thread involuntary context switches"),
                counter_delta(timing->end.process_nvcsw, timing->start.process_nvcsw,
                    "process voluntary context switches"),
                counter_delta(timing->end.process_nivcsw, timing->start.process_nivcsw,
                    "process involuntary context switches"),
                bytes_per_dpu,
                bytes_per_dpu * actual_dpus,
                result_ok ? 1U : 0U);
            write_csv_field(stream, env_or("VA_SOURCE_SHA256", "unknown"));
            fputc(',', stream);
            write_csv_field(stream, env_or("VA_HOST_BINARY_SHA256", "unknown"));
            fputc(',', stream);
            write_csv_field(stream, env_or("VA_DPU_BINARY_SHA256", "unknown"));
            fputc('\n', stream);
        }
    }
    if (fclose(stream) != 0) {
        perror("fclose VA transfer trace");
        exit(EXIT_FAILURE);
    }
}

static void
read_input(T *input_a, T *input_b, unsigned int nr_elements)
{
#if VA_VALIDATION_INPUT
    for (unsigned int i = 0; i < nr_elements; ++i) {
        input_a[i] = (T)i;
        input_b[i] = (T)i;
    }
#else
    srand(0);
    for (unsigned int i = 0; i < nr_elements; ++i) {
        input_a[i] = (T)rand();
        input_b[i] = (T)rand();
    }
#endif
}

static void
vector_addition_host(T *output, T *input_a, T *input_b, unsigned int nr_elements)
{
    for (unsigned int i = 0; i < nr_elements; ++i)
        output[i] = input_a[i] + input_b[i];
}

static void
prepare_and_push_mram(
    struct dpu_set_t dpu_set,
    T *buffer,
    unsigned int per_dpu_elements,
    unsigned int mram_offset,
    enum transfer_component_id prepare_component,
    enum transfer_component_id push_component,
    struct transfer_component_timing *sample_timings,
    struct timing_snapshot *boundary)
{
    struct dpu_set_t dpu;
    uint32_t i = 0;

    if (sample_timings != NULL)
        sample_timings[prepare_component].start = *boundary;
    DPU_FOREACH(dpu_set, dpu, i)
        DPU_ASSERT(dpu_prepare_xfer(dpu, buffer + per_dpu_elements * i));
    if (sample_timings != NULL) {
        *boundary = take_timing_snapshot();
        sample_timings[prepare_component].end = *boundary;
        sample_timings[push_component].start = *boundary;
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
        DPU_MRAM_HEAP_POINTER_NAME, mram_offset,
        per_dpu_elements * sizeof(T), DPU_XFER_DEFAULT));
    if (sample_timings != NULL) {
        *boundary = take_timing_snapshot();
        sample_timings[push_component].end = *boundary;
    }
}

int
main(int argc, char **argv)
{
    struct Params p = input_params(argc, argv);
    struct dpu_set_t dpu_set, dpu;
    uint32_t nr_of_dpus = 0;
    uint32_t nr_of_ranks = 0;
    const char *requested_mode = getenv("VA_ALLOCATION_MODE");
    const char *configuration = requested_mode == NULL ? "configured" : requested_mode;
    const char *profile = getenv("VA_DPU_PROFILE");
    const char *transfer_order = env_or("VA_TRANSFER_ORDER", "AB");
    const char *trace_path = getenv("VA_TRACE_CSV");
    const char *transfer_trace_path = getenv("VA_TRANSFER_BREAKDOWN_CSV");
    bool transfer_trace_enabled =
        transfer_trace_path != NULL && transfer_trace_path[0] != '\0';
    uint64_t setup_start_ns = raw_time_ns();

    if (strcmp(transfer_order, "AB") != 0 && strcmp(transfer_order, "BA") != 0) {
        fprintf(stderr, "VA_TRANSFER_ORDER must be AB or BA\n");
        return EXIT_FAILURE;
    }
    if (profile != NULL && profile[0] == '\0')
        profile = NULL;
    if (strcmp(configuration, "single") == 0) {
        DPU_ASSERT(dpu_alloc(1, profile, &dpu_set));
    } else if (strcmp(configuration, "rank") == 0) {
        DPU_ASSERT(dpu_alloc_ranks(1, profile, &dpu_set));
    } else if (strcmp(configuration, "configured") == 0) {
        DPU_ASSERT(dpu_alloc(NR_DPUS, profile, &dpu_set));
    } else {
        fprintf(stderr, "VA_ALLOCATION_MODE must be single or rank\n");
        return EXIT_FAILURE;
    }
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));
    DPU_ASSERT(dpu_get_nr_ranks(dpu_set, &nr_of_ranks));
    uint64_t setup_ns = raw_time_ns() - setup_start_ns;

    if ((strcmp(configuration, "single") == 0 && nr_of_dpus != 1) ||
        (strcmp(configuration, "rank") == 0 && nr_of_ranks != 1)) {
        fprintf(stderr, "allocated topology differs from VA validation mode\n");
        DPU_ASSERT(dpu_free(dpu_set));
        return EXIT_FAILURE;
    }

#if ENERGY
    struct dpu_probe_t probe;
    DPU_ASSERT(dpu_probe_init("energy_probe", &probe));
#endif

    const unsigned int input_size = p.exp == 0 ? p.input_size * nr_of_dpus : p.input_size;
    const unsigned int input_size_8bytes =
        ((input_size * sizeof(T)) % 8) != 0 ? roundup(input_size, 8) : input_size;
    const unsigned int input_size_dpu = divceil(input_size, nr_of_dpus);
    const unsigned int input_size_dpu_8bytes =
        ((input_size_dpu * sizeof(T)) % 8) != 0 ? roundup(input_size_dpu, 8) : input_size_dpu;

    A = calloc((size_t)input_size_dpu_8bytes * nr_of_dpus, sizeof(T));
    B = calloc((size_t)input_size_dpu_8bytes * nr_of_dpus, sizeof(T));
    C = calloc((size_t)input_size_dpu_8bytes * nr_of_dpus, sizeof(T));
    C2 = calloc((size_t)input_size_dpu_8bytes * nr_of_dpus, sizeof(T));
    dpu_arguments_t *input_arguments = calloc(nr_of_dpus, sizeof(*input_arguments));
    uint64_t *cpu_ns = calloc((size_t)p.n_reps, sizeof(*cpu_ns));
    uint64_t *h2d_ns = calloc((size_t)p.n_reps, sizeof(*h2d_ns));
    uint64_t *kernel_ns = calloc((size_t)p.n_reps, sizeof(*kernel_ns));
    uint64_t *d2h_ns = calloc((size_t)p.n_reps, sizeof(*d2h_ns));
    struct transfer_component_timing *transfer_timings = transfer_trace_enabled
        ? calloc((size_t)p.n_reps * TRANSFER_COMPONENT_COUNT, sizeof(*transfer_timings))
        : NULL;
    if (A == NULL || B == NULL || C == NULL || C2 == NULL || input_arguments == NULL ||
        cpu_ns == NULL || h2d_ns == NULL || kernel_ns == NULL || d2h_ns == NULL ||
        (transfer_trace_enabled && transfer_timings == NULL)) {
        fprintf(stderr, "VA host allocation failed\n");
        return EXIT_FAILURE;
    }
    T *bufferA = A;
    T *bufferB = B;
    T *bufferC = C2;

    read_input(A, B, input_size);
    printf("Allocated %u DPU(s) across %u rank(s)\n", nr_of_dpus, nr_of_ranks);
    printf("NR_TASKLETS\t%d\tBL\t%d\tINPUT_ELEMENTS\t%u\n",
        NR_TASKLETS, BL, input_size);
    printf("VA_TRANSFER_ORDER\t%s\n", transfer_order);

    for (uint32_t i = 0; i < nr_of_dpus - 1; ++i) {
        input_arguments[i].size = input_size_dpu_8bytes * sizeof(T);
        input_arguments[i].transfer_size = input_size_dpu_8bytes * sizeof(T);
        input_arguments[i].kernel = kernel1;
    }
    input_arguments[nr_of_dpus - 1].size =
        (input_size_8bytes - input_size_dpu_8bytes * (nr_of_dpus - 1)) * sizeof(T);
    input_arguments[nr_of_dpus - 1].transfer_size = input_size_dpu_8bytes * sizeof(T);
    input_arguments[nr_of_dpus - 1].kernel = kernel1;

    for (int rep = 0; rep < p.n_warmup + p.n_reps; ++rep) {
        bool measured = rep >= p.n_warmup;
        int sample = rep - p.n_warmup;
        uint64_t phase_start = raw_time_ns();
        vector_addition_host(C, A, B, input_size);
        if (measured)
            cpu_ns[sample] = raw_time_ns() - phase_start;

        struct timing_snapshot boundary;
        struct transfer_component_timing *sample_timings = transfer_trace_enabled && measured
            ? &transfer_timings[(size_t)sample * TRANSFER_COMPONENT_COUNT]
            : NULL;

        phase_start = raw_time_ns();
        if (sample_timings != NULL) {
            boundary = take_timing_snapshot();
            phase_start = boundary.wall_ns;
            sample_timings[COMPONENT_PREPARE_ARGS].start = boundary;
        }
        uint32_t i = 0;
        DPU_FOREACH(dpu_set, dpu, i)
            DPU_ASSERT(dpu_prepare_xfer(dpu, &input_arguments[i]));
        if (sample_timings != NULL) {
            boundary = take_timing_snapshot();
            sample_timings[COMPONENT_PREPARE_ARGS].end = boundary;
            sample_timings[COMPONENT_PUSH_ARGS].start = boundary;
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS",
            0, sizeof(input_arguments[0]), DPU_XFER_DEFAULT));
        if (sample_timings != NULL) {
            boundary = take_timing_snapshot();
            sample_timings[COMPONENT_PUSH_ARGS].end = boundary;
        }

        if (strcmp(transfer_order, "AB") == 0) {
            prepare_and_push_mram(dpu_set, bufferA, input_size_dpu_8bytes, 0,
                COMPONENT_PREPARE_A, COMPONENT_PUSH_A, sample_timings, &boundary);
            prepare_and_push_mram(dpu_set, bufferB, input_size_dpu_8bytes,
                input_size_dpu_8bytes * sizeof(T), COMPONENT_PREPARE_B,
                COMPONENT_PUSH_B, sample_timings, &boundary);
        } else {
            prepare_and_push_mram(dpu_set, bufferB, input_size_dpu_8bytes,
                input_size_dpu_8bytes * sizeof(T), COMPONENT_PREPARE_B,
                COMPONENT_PUSH_B, sample_timings, &boundary);
            prepare_and_push_mram(dpu_set, bufferA, input_size_dpu_8bytes, 0,
                COMPONENT_PREPARE_A, COMPONENT_PUSH_A, sample_timings, &boundary);
        }
        if (measured) {
            if (sample_timings != NULL) {
                h2d_ns[sample] = boundary.wall_ns - phase_start;
            } else {
                h2d_ns[sample] = raw_time_ns() - phase_start;
            }
        }

        phase_start = raw_time_ns();
#if ENERGY
        if (measured)
            DPU_ASSERT(dpu_probe_start(&probe));
#endif
        DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
#if ENERGY
        if (measured)
            DPU_ASSERT(dpu_probe_stop(&probe));
#endif
        if (measured)
            kernel_ns[sample] = raw_time_ns() - phase_start;

        phase_start = raw_time_ns();
        if (sample_timings != NULL) {
            boundary = take_timing_snapshot();
            phase_start = boundary.wall_ns;
            sample_timings[COMPONENT_PREPARE_C].start = boundary;
        }
        DPU_FOREACH(dpu_set, dpu, i)
            DPU_ASSERT(dpu_prepare_xfer(dpu, bufferC + input_size_dpu_8bytes * i));
        if (sample_timings != NULL) {
            boundary = take_timing_snapshot();
            sample_timings[COMPONENT_PREPARE_C].end = boundary;
            sample_timings[COMPONENT_PUSH_C].start = boundary;
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU,
            DPU_MRAM_HEAP_POINTER_NAME, input_size_dpu_8bytes * sizeof(T),
            input_size_dpu_8bytes * sizeof(T), DPU_XFER_DEFAULT));
        if (measured) {
            if (sample_timings != NULL) {
                boundary = take_timing_snapshot();
                sample_timings[COMPONENT_PUSH_C].end = boundary;
                d2h_ns[sample] = boundary.wall_ns - phase_start;
            } else {
                d2h_ns[sample] = raw_time_ns() - phase_start;
            }
        }
    }

    uint64_t verify_start_ns = raw_time_ns();
    bool status = true;
    uint64_t expected_checksum = 0;
    uint64_t actual_checksum = 0;
    for (unsigned int i = 0; i < input_size; ++i) {
        expected_checksum += (uint64_t)(uint32_t)C[i];
        actual_checksum += (uint64_t)(uint32_t)bufferC[i];
        if (C[i] != bufferC[i])
            status = false;
    }
    uint64_t verify_ns = raw_time_ns() - verify_start_ns;

    double cpu_ms = 0.0;
    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
    for (int rep = 0; rep < p.n_reps; ++rep) {
        cpu_ms += (double)cpu_ns[rep] / 1.0e6;
        h2d_ms += (double)h2d_ns[rep] / 1.0e6;
        kernel_ms += (double)kernel_ns[rep] / 1.0e6;
        d2h_ms += (double)d2h_ns[rep] / 1.0e6;
    }
    printf("CPU Time (ms): %f\t", cpu_ms / p.n_reps);
    printf("CPU-DPU Time (ms): %f\t", h2d_ms / p.n_reps);
    printf("DPU Kernel Time (ms): %f\t", kernel_ms / p.n_reps);
    printf("DPU-CPU Time (ms): %f\n", d2h_ms / p.n_reps);
    printf("VA_CHECKSUM expected=%" PRIu64 " actual=%" PRIu64 "\n",
        expected_checksum, actual_checksum);
    printf(status ? "[OK] Outputs are equal\n" : "[ERROR] Outputs differ!\n");

    if (trace_path != NULL && trace_path[0] != '\0') {
        write_trace(trace_path, configuration, transfer_order, nr_of_dpus, nr_of_ranks,
            input_size, input_size_dpu_8bytes, p.n_reps, cpu_ns, h2d_ns,
            kernel_ns, d2h_ns, setup_ns, verify_ns, status,
            expected_checksum, actual_checksum);
    }
    if (transfer_trace_enabled) {
        write_transfer_breakdown(transfer_trace_path, configuration, nr_of_dpus,
            nr_of_ranks, input_size, input_size_dpu_8bytes, p.n_reps,
            transfer_timings, status);
    }

    free(transfer_timings);
    free(cpu_ns);
    free(h2d_ns);
    free(kernel_ns);
    free(d2h_ns);
    free(input_arguments);
    free(A);
    free(B);
    free(C);
    free(C2);
    DPU_ASSERT(dpu_free(dpu_set));

    return status ? EXIT_SUCCESS : EXIT_FAILURE;
}
