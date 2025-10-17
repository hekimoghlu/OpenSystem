/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 8, 2022.
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
#include <pthread.h>
#include <stdbool.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <mach/mach_time.h>
#include <dispatch/dispatch.h>

#include <darwintest.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.workq"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("workq"),
	T_META_RUN_CONCURRENTLY(true));


static pthread_t workq_thread;
static bool signal_received;

static void
signal_handler(int sig __unused, siginfo_t *b __unused, void* unused __unused)
{
	if (pthread_self() == workq_thread) {
		signal_received = true;
	}
}

static void
workq_block(void *unused __unused)
{
	workq_thread = pthread_self();

	/*
	 *  sigset_t set;
	 *  sigemptyset(&set);
	 *  sigaddset(&set, SIGPROF);
	 *  pthread_sigmask(SIG_UNBLOCK, &set, NULL);
	 */

	uint64_t spin_start = mach_absolute_time();
	while (mach_absolute_time() - spin_start < 30 * NSEC_PER_SEC) {
		if (signal_received) {
			T_PASS("Got SIGPROF!");
			T_END;
		}
	}
}

T_DECL(workq_sigprof, "test that workqueue threads can receive sigprof", T_META_TAG_VM_PREFERRED)
{
	struct sigaction sa = {
		.sa_sigaction = signal_handler
	};
	sigfillset(&sa.sa_mask);
	T_ASSERT_POSIX_ZERO(sigaction(SIGPROF, &sa, NULL), NULL);

	dispatch_queue_t q = dispatch_get_global_queue(0, 0);
	dispatch_async_f(q, NULL, workq_block);

	struct itimerval timerval = {
		.it_interval = {.tv_usec = 10000},
		.it_value = {.tv_usec = 10000}
	};
	T_ASSERT_POSIX_ZERO(setitimer(ITIMER_PROF, &timerval, NULL), NULL);

	dispatch_main();
}
