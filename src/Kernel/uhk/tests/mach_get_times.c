/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 19, 2024.
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
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <mach/mach_time.h>

#include <darwintest.h>
#include <darwintest_utils.h>

T_GLOBAL_META(T_META_RUN_CONCURRENTLY(true));

#define T_LOG_VERBOSE(...)

#define timespec2nanosec(ts) ((uint64_t)((ts)->tv_sec) * NSEC_PER_SEC + (uint64_t)((ts)->tv_nsec))

T_DECL(mach_get_times, "mach_get_times()",
    T_META_CHECK_LEAKS(false), T_META_ALL_VALID_ARCHS(true))
{
	const int ITERATIONS = 500000 * dt_ncpu();
	struct timespec gtod_ts;

	uint64_t last_absolute, last_continuous, last_gtod;
	T_QUIET; T_ASSERT_EQ(mach_get_times(&last_absolute, &last_continuous, &gtod_ts), KERN_SUCCESS, NULL);
	last_gtod = timespec2nanosec(&gtod_ts);

	for (int i = 0; i < ITERATIONS; i++) {
		uint64_t absolute, continuous, gtod;
		T_QUIET; T_ASSERT_EQ(mach_get_times(&absolute, &continuous, &gtod_ts), KERN_SUCCESS, NULL);
		gtod = timespec2nanosec(&gtod_ts);

		T_LOG_VERBOSE("[%d] abs: %llu.%09llu(+%llu)\tcont: %llu.%09llu(+%llu)\tgtod:%llu.%09llu(+%llu)", i,
		    absolute / NSEC_PER_SEC, absolute % NSEC_PER_SEC, absolute - last_absolute,
		    continuous / NSEC_PER_SEC, continuous % NSEC_PER_SEC, continuous - last_continuous,
		    gtod / NSEC_PER_SEC, gtod % NSEC_PER_SEC, gtod - last_gtod);

		T_QUIET; T_EXPECT_EQ(absolute - last_absolute, continuous - last_continuous, NULL);

		int64_t gtod_diff = (int64_t)gtod - (int64_t)last_gtod;
		T_QUIET; T_ASSERT_LE((uint64_t)llabs(gtod_diff), NSEC_PER_SEC, NULL);

		last_absolute = absolute;
		last_continuous = continuous;
		last_gtod = gtod;

		gtod_ts.tv_sec = 0; gtod_ts.tv_nsec = 0;
	}
}
