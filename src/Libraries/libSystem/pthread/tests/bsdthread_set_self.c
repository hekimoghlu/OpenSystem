/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 25, 2025.
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

#include <assert.h>
#include <errno.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#include <sys/qos.h>

#include <pthread.h>
#include <pthread/tsd_private.h>
#include <pthread/qos_private.h>
#include <pthread/workqueue_private.h>

#include <dispatch/dispatch.h>

#include "darwintest_defaults.h"

T_DECL(bsdthread_set_self_constrained_transition, "bsdthread_ctl(SET_SELF) with overcommit change",
		T_META_ALL_VALID_ARCHS(YES))
{
	dispatch_async(dispatch_get_global_queue(0, 0), ^{
		pthread_priority_t overcommit = (pthread_priority_t)_pthread_getspecific_direct(_PTHREAD_TSD_SLOT_PTHREAD_QOS_CLASS) |
			_PTHREAD_PRIORITY_OVERCOMMIT_FLAG;
		pthread_priority_t constrained = overcommit & (~_PTHREAD_PRIORITY_OVERCOMMIT_FLAG);

		T_ASSERT_POSIX_ZERO(_pthread_set_properties_self(_PTHREAD_SET_SELF_QOS_FLAG, overcommit, 0), NULL);
		T_ASSERT_POSIX_ZERO(_pthread_set_properties_self(_PTHREAD_SET_SELF_QOS_FLAG, constrained, 0), NULL);
		T_ASSERT_POSIX_ZERO(_pthread_set_properties_self(_PTHREAD_SET_SELF_QOS_FLAG, overcommit, 0), NULL);
		T_ASSERT_POSIX_ZERO(_pthread_set_properties_self(_PTHREAD_SET_SELF_QOS_FLAG, constrained, 0), NULL);
		T_ASSERT_POSIX_ZERO(_pthread_set_properties_self(_PTHREAD_SET_SELF_QOS_FLAG, overcommit, 0), NULL);
		T_ASSERT_POSIX_ZERO(_pthread_set_properties_self(_PTHREAD_SET_SELF_QOS_FLAG, constrained, 0), NULL);

		T_END;
	});

	dispatch_main();
}

T_DECL(bsdthread_set_self_constrained_threads, "bsdthread_ctl(SET_SELF) with overcommit change",
		T_META_CHECK_LEAKS(NO), T_META_ALL_VALID_ARCHS(YES))
{
	static const int THREADS = 128;
	static atomic_int threads_started;
	dispatch_queue_t q = dispatch_queue_create("my queue", DISPATCH_QUEUE_CONCURRENT);
	dispatch_set_target_queue(q, dispatch_get_global_queue(0, 0));
	for (int i = 0; i < THREADS; i++) {
		dispatch_async(q, ^{
			int thread_id = ++threads_started;
			T_PASS("Thread %d started successfully", thread_id);
			if (thread_id == THREADS){
				T_PASS("All threads started successfully");
				T_END;
			}

			pthread_priority_t overcommit = (pthread_priority_t)_pthread_getspecific_direct(_PTHREAD_TSD_SLOT_PTHREAD_QOS_CLASS) |
				_PTHREAD_PRIORITY_OVERCOMMIT_FLAG;
			T_ASSERT_POSIX_ZERO(_pthread_set_properties_self(_PTHREAD_SET_SELF_QOS_FLAG, overcommit, 0), NULL);

			uint64_t t = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW);
			while (t > clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW) - (thread_id == 1 ? 30 : 60) * NSEC_PER_SEC) {
				sleep(1);
			}
			if (thread_id == 1) {
				T_FAIL("Where are my threads?");
				T_END;
			}
		});
	}

	dispatch_main();
}
