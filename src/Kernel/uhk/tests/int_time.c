/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 11, 2024.
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
#include <darwintest.h>
#include <darwintest_utils.h>
#include <errno.h>
#include <libproc_internal.h>
#include <stdio.h>

// waste some time.
static uint64_t
ackermann(uint64_t m, uint64_t n)
{
	if (m == 0) {
		return n + 1;
	} else if (n == 0) {
		return ackermann(m - 1, 1);
	} else {
		return ackermann(m - 1, ackermann(m, n - 1));
	}
}

T_DECL(int_time, "interrupt time collection")
{
	struct proc_threadschedinfo info, info_new;

	// for checking that the kernel filled in values.
	info.int_time_ns = (uint64_t)-3;
	info_new.int_time_ns = (uint64_t)-3;

	int retval = proc_current_thread_schedinfo((void*)&info, sizeof(info));
	T_ASSERT_EQ_INT(retval, 0, "proc_current_thread_schedinfo succeeded");
	T_ASSERT_NE(info.int_time_ns, (uint64_t)-3, "info.int_time_ns was filled");

	printf("result: %llu\n", ackermann(3, 10));

	retval = proc_current_thread_schedinfo((void*)&info_new, sizeof(info_new));
	T_ASSERT_EQ_INT(retval, 0, "proc_current_thread_schedinfo succeeded (2nd time)");
	T_ASSERT_NE(info_new.int_time_ns, (uint64_t)-3, "info.int_time_ns was filled (2nd time)");

	int64_t duration_ns = (int64_t)info_new.int_time_ns - (int64_t)info.int_time_ns;

	printf("before   : %lluns\n",
	    info.int_time_ns);
	printf("after    : %lluns\n",
	    info_new.int_time_ns);
	printf("duration : %llins\n", duration_ns);

	/*
	 * If time went backwards, life is missing its monotony.
	 */

	T_EXPECT_FALSE(duration_ns < 0, "Kernel claims time went forewards");

	/*
	 * If the kernel says we spent more than 10 seconds in an interrupt context, something is definitely wrong.
	 */

	int64_t const limit_ns = 10 * (int64_t)NSEC_PER_SEC;
	T_EXPECT_FALSE(duration_ns > limit_ns, "Kernel claims we spent less than 10 seconds in interrupt context");
}
