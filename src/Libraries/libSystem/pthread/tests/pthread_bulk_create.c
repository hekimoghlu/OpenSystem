/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 13, 2023.
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

#define MAX_THREADS 512
#define THREAD_DEPTH 32

static void *
thread(void * arg)
{
	T_LOG("thread %lx here: %d", (uintptr_t)pthread_self(), (int)arg);
	return (arg);
}

T_DECL(pthread_bulk_create, "pthread_bulk_create")
{
	void *thread_res;
	pthread_t t[THREAD_DEPTH];

	for (int i = 0; i < MAX_THREADS; i += THREAD_DEPTH) {
		T_LOG("Creating threads %d..%d\n", i, i + THREAD_DEPTH - 1);
		for (int j = 0; j < THREAD_DEPTH; j++) {
			void *arg = (void *)(intptr_t)(i + j);
			T_QUIET; T_ASSERT_POSIX_ZERO(
					pthread_create(&t[j], NULL, thread, arg), NULL);
		}
		T_LOG("Waiting for threads");
		for (int j = 0; j < THREAD_DEPTH; j++) {
			T_QUIET; T_ASSERT_POSIX_ZERO(pthread_join(t[j], &thread_res), NULL);
			T_QUIET; T_ASSERT_EQ(i + j, (int)thread_res, "thread return value");
		}
	}
}
