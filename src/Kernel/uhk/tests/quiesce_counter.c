/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 3, 2022.
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
/*
 * Test to validate that _COMM_PAGE_CPU_QUIESCENT_COUNTER ticks at least once per second
 *
 * <rdar://problem/42433973>
 */

#include <System/machine/cpu_capabilities.h>

#include <darwintest.h>

#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/sysctl.h>

T_GLOBAL_META(T_META_RUN_CONCURRENTLY(true));

#ifndef _COMM_PAGE_CPU_QUIESCENT_COUNTER

T_DECL(test_quiescent_counter, "Validate that _COMM_PAGE_CPU_QUIESCENT_COUNTER increments",
    T_META_CHECK_LEAKS(false))
{
	T_SKIP("_COMM_PAGE_CPU_QUIESCENT_COUNTER doesn't exist on this system");
}

#else /* _COMM_PAGE_CPU_QUIESCENT_COUNTER */

T_DECL(test_quiescent_counter, "Validate that _COMM_PAGE_CPU_QUIESCENT_COUNTER increments",
    T_META_CHECK_LEAKS(false))
{
	int rv;

	uint32_t cpu_checkin_min_interval = 0; /* set by sysctl hw.ncpu */

	size_t value_size = sizeof(cpu_checkin_min_interval);
	rv = sysctlbyname("kern.cpu_checkin_interval", &cpu_checkin_min_interval, &value_size, NULL, 0);
	T_ASSERT_POSIX_SUCCESS(rv, "sysctlbyname(kern.cpu_checkin_interval)");

	T_LOG("kern.cpu_checkin_interval is %d", cpu_checkin_min_interval);

	T_ASSERT_GT(cpu_checkin_min_interval, 0, "kern.cpu_checkin_interval should be > 0");

	COMM_PAGE_SLOT_TYPE(uint64_t) commpage_addr = COMM_PAGE_SLOT(uint64_t, CPU_QUIESCENT_COUNTER);

	T_LOG("address of _COMM_PAGE_CPU_QUIESCENT_COUNTER is %p", commpage_addr);

	uint64_t counter = *commpage_addr;
	uint64_t last_counter = counter;
	T_LOG("first value of _COMM_PAGE_CPU_QUIESCENT_COUNTER is %llu", counter);

	for (int i = 0; i < 10; i++) {
		sleep(1);

		last_counter = counter;
		counter = *commpage_addr;

		T_LOG("value of _COMM_PAGE_CPU_QUIESCENT_COUNTER is %llu", counter);

		T_ASSERT_GT(counter, last_counter, "_COMM_PAGE_CPU_QUIESCENT_COUNTER must monotonically increase at least once per second");
	}
}

#endif /* _COMM_PAGE_CPU_QUIESCENT_COUNTER */
