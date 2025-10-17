/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 13, 2022.
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

#include <sys/sysctl.h>
#include <dispatch/dispatch.h>
#include <dispatch/private.h>
#include "darwintest_defaults.h"

T_DECL(wq_pool_limits, "test overcommit limit")
{
	dispatch_semaphore_t sema = dispatch_semaphore_create(0);
	dispatch_group_t g = dispatch_group_create();
	dispatch_time_t t;
	uint32_t wq_max_threads, wq_max_constrained_threads;

	dispatch_block_t b = ^{
		dispatch_group_leave(g);
		dispatch_semaphore_wait(sema, DISPATCH_TIME_FOREVER);
	};

	size_t s = sizeof(uint32_t);
	sysctlbyname("kern.wq_max_threads", &wq_max_threads, &s, NULL, 0);
	sysctlbyname("kern.wq_max_constrained_threads", &wq_max_constrained_threads,
				 &s, NULL, 0);

	for (uint32_t i = 0; i < wq_max_constrained_threads; i++) {
		dispatch_group_enter(g);
		dispatch_async(dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), b);
	}

	t = dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC);
	T_ASSERT_EQ(dispatch_group_wait(g, t), 0L,
			"%d constrained threads bringup", wq_max_constrained_threads);

	dispatch_async(dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^{
		T_ASSERT_FAIL("Should never run");
	});

	sleep(5);
	T_PASS("constrained limit looks fine");

	for (uint32_t i = wq_max_constrained_threads; i < wq_max_threads; i++) {
		dispatch_group_enter(g);
		dispatch_async(dispatch_get_global_queue(QOS_CLASS_DEFAULT,
						DISPATCH_QUEUE_OVERCOMMIT), b);
	}
	t = dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC);
	T_ASSERT_EQ(dispatch_group_wait(g, t), 0L,
			"%d threads bringup", wq_max_threads);


	dispatch_async(dispatch_get_global_queue(QOS_CLASS_DEFAULT,
					DISPATCH_QUEUE_OVERCOMMIT), ^{
		T_ASSERT_FAIL("Should never run");
	});

	sleep(5);
	T_PASS("thread limit looks fine");
	T_END;
}
