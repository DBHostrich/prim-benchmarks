#define _GNU_SOURCE

#include <errno.h>
#include <inttypes.h>
#include <sched.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t stop_requested = 0;

static void handle_signal(int signal_number) {
	(void)signal_number;
	stop_requested = 1;
}

static uint64_t timespec_ns(const struct timespec *value) {
	return (uint64_t)value->tv_sec * UINT64_C(1000000000)
		+ (uint64_t)value->tv_nsec;
}

static struct timespec ns_timespec(uint64_t value) {
	struct timespec result;
	result.tv_sec = (time_t)(value / UINT64_C(1000000000));
	result.tv_nsec = (long)(value % UINT64_C(1000000000));
	return result;
}

static long parse_long(const char *name, const char *value, long minimum) {
	char *end = NULL;
	long parsed;
	errno = 0;
	parsed = strtol(value, &end, 10);
	if (errno != 0 || end == value || *end != '\0' || parsed < minimum) {
		fprintf(stderr, "Invalid %s: %s\n", name, value);
		exit(EXIT_FAILURE);
	}
	return parsed;
}

static void pin_to_cpu(int cpu_id) {
	cpu_set_t set;
	CPU_ZERO(&set);
	CPU_SET(cpu_id, &set);
	if (sched_setaffinity(0, sizeof(set), &set) != 0) {
		perror("sched_setaffinity");
		exit(EXIT_FAILURE);
	}
}

int main(int argc, char **argv) {
	const char *output_path = NULL;
	long cpu_id = -1;
	long period_us = 100;
	long threshold_us = 50;
	uint64_t period_ns;
	uint64_t threshold_ns;
	uint64_t next_ns;
	uint64_t samples = 0;
	FILE *stream;
	struct sigaction action = {0};
	int option;

	while ((option = getopt(argc, argv, "c:o:p:t:")) != -1) {
		switch (option) {
		case 'c':
			cpu_id = parse_long("CPU ID", optarg, 0);
			break;
		case 'o':
			output_path = optarg;
			break;
		case 'p':
			period_us = parse_long("period", optarg, 10);
			break;
		case 't':
			threshold_us = parse_long("threshold", optarg, 1);
			break;
		default:
			fprintf(stderr,
				"Usage: %s -c CPU -o OUTPUT [-p PERIOD_US] "
				"[-t THRESHOLD_US]\n", argv[0]);
			return EXIT_FAILURE;
		}
	}
	if (cpu_id < 0 || output_path == NULL || optind != argc) {
		fprintf(stderr,
			"Usage: %s -c CPU -o OUTPUT [-p PERIOD_US] "
			"[-t THRESHOLD_US]\n", argv[0]);
		return EXIT_FAILURE;
	}
	if ((uint64_t)period_us > UINT64_MAX / UINT64_C(1000)) {
		fprintf(stderr, "Heartbeat period overflows nanoseconds\n");
		return EXIT_FAILURE;
	}
	period_ns = (uint64_t)period_us * UINT64_C(1000);
	threshold_ns = (uint64_t)threshold_us * UINT64_C(1000);
	pin_to_cpu((int)cpu_id);

	action.sa_handler = handle_signal;
	sigemptyset(&action.sa_mask);
	if (sigaction(SIGINT, &action, NULL) != 0
		|| sigaction(SIGTERM, &action, NULL) != 0) {
		perror("sigaction");
		return EXIT_FAILURE;
	}
	stream = fopen(output_path, "w");
	if (stream == NULL) {
		perror(output_path);
		return EXIT_FAILURE;
	}
	fputs(
		"planned_raw_ns,actual_raw_ns,actual_mono_ns,lateness_ns,cpu_id\n",
		stream
	);
	fflush(stream);

	{
		struct timespec now;
		if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
			perror("clock_gettime(CLOCK_MONOTONIC)");
			fclose(stream);
			return EXIT_FAILURE;
		}
		next_ns = timespec_ns(&now) + period_ns;
	}

	while (!stop_requested) {
		struct timespec deadline = ns_timespec(next_ns);
		struct timespec actual_mono;
		struct timespec actual_raw;
		uint64_t actual_mono_ns;
		uint64_t actual_raw_ns;
		uint64_t planned_raw_ns;
		uint64_t lateness_ns;
		int64_t raw_minus_mono;
		int status;

		do {
			status = clock_nanosleep(
				CLOCK_MONOTONIC, TIMER_ABSTIME, &deadline, NULL
			);
		} while (status == EINTR && !stop_requested);
		if (stop_requested)
			break;
		if (status != 0) {
			errno = status;
			perror("clock_nanosleep");
			fclose(stream);
			return EXIT_FAILURE;
		}
		if (clock_gettime(CLOCK_MONOTONIC, &actual_mono) != 0
			|| clock_gettime(CLOCK_MONOTONIC_RAW, &actual_raw) != 0) {
			perror("clock_gettime");
			fclose(stream);
			return EXIT_FAILURE;
		}
		actual_mono_ns = timespec_ns(&actual_mono);
		actual_raw_ns = timespec_ns(&actual_raw);
		raw_minus_mono = (int64_t)actual_raw_ns - (int64_t)actual_mono_ns;
		planned_raw_ns = raw_minus_mono >= 0
			? next_ns + (uint64_t)raw_minus_mono
			: next_ns - (uint64_t)(-raw_minus_mono);
		lateness_ns = actual_mono_ns > next_ns ? actual_mono_ns - next_ns : 0;
		if (lateness_ns >= threshold_ns) {
			int actual_cpu = sched_getcpu();
			if (actual_cpu < 0) {
				perror("sched_getcpu");
				fclose(stream);
				return EXIT_FAILURE;
			}
			fprintf(stream,
				"%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
				",%d\n",
				planned_raw_ns, actual_raw_ns, actual_mono_ns, lateness_ns,
				actual_cpu);
		}
		++samples;
		if (samples % UINT64_C(1000) == 0)
			fflush(stream);
		do {
			next_ns += period_ns;
		} while (next_ns <= actual_mono_ns);
	}
	if (fclose(stream) != 0) {
		perror("fclose");
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
