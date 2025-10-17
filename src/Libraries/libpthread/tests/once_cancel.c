/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 26, 2024.
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

#include "darwintest_defaults.h"

static volatile int once_invoked = 0;

static void
cancelation_handler(void * __unused arg)
{
	T_LOG("cancelled");
}

__attribute__((noreturn))
static void
await_cancelation(void)
{
	pthread_cleanup_push(cancelation_handler, NULL);
	T_LOG("waiting for cancellation");

	// can't use darwintest once cancellation is enabled
	pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);

	while (true) {
		pthread_testcancel();
		sched_yield();
	}

	pthread_cleanup_pop(0);
}

static void *
await_cancelation_in_once(void *arg)
{
	// disable cancellation until pthread_once to protect darwintest
	pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);

	T_LOG("starting the thread");
	pthread_once_t *once = (pthread_once_t *)arg;
	pthread_once(once, await_cancelation);
	return NULL;
}

static void
oncef(void)
{
	T_LOG("once invoked");
	once_invoked++;
}

T_DECL(once_cancel, "pthread_once is re-executed if cancelled")
{
	pthread_once_t once = PTHREAD_ONCE_INIT;
	pthread_t t;
	void *join_result = NULL;

	T_ASSERT_POSIX_ZERO(
			pthread_create(&t, NULL, await_cancelation_in_once, &once), NULL);
	T_ASSERT_POSIX_ZERO(pthread_cancel(t), NULL);
	T_ASSERT_POSIX_ZERO(pthread_join(t, &join_result), NULL);
	T_ASSERT_EQ(join_result, PTHREAD_CANCELED, NULL);

	T_ASSERT_POSIX_ZERO(pthread_once(&once, oncef), NULL);
	T_ASSERT_EQ(once_invoked, 1, NULL);
}
