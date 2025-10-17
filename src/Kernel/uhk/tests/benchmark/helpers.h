/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 7, 2023.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */

#ifndef BENCHMARK_PERF_HELPERS_H
#define BENCHMARK_PERF_HELPERS_H

/*
 * Utility functions and constants used by perf tests.
 */
#include <inttypes.h>
#include <time.h>
#include <stdbool.h>

/*
 * mmap an anonymous chunk of memory.
 */
unsigned char *map_buffer(size_t size, int flags);
/*
 * Returns a - b in microseconds.
 * NB: a must be >= b
 */
uint64_t timespec_difference_us(const struct timespec* a, const struct timespec* b);
/*
 * Print the message to stdout along with the current time.
 * Also flushes stdout so that the log can help detect hangs. Don't call
 * this function from within the measured portion of the benchmark as it will
 * pollute your measurement.
 *
 * NB: Will only log if verbose == true.
 */
void benchmark_log(bool verbose, const char *restrict fmt, ...) __attribute__((format(printf, 2, 3)));

static const uint64_t kNumMicrosecondsInSecond = 1000UL * 1000;
static const uint64_t kNumNanosecondsInMicrosecond = 1000UL;
static const uint64_t kNumNanosecondsInSecond = kNumNanosecondsInMicrosecond * kNumMicrosecondsInSecond;
/* Get a (wall-time) timestamp in nanoseconds */
#define current_timestamp_ns() (clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW));

unsigned int get_ncpu(void);

#endif /* !defined(BENCHMARK_PERF_HELPERS_H) */
