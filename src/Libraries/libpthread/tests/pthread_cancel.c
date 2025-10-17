/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 10, 2024.
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

static void *
run(void * __unused arg)
{
	while (true) {
		pthread_testcancel();
		sched_yield();
	}
}

T_DECL(pthread_cancel, "pthread_cancel",
		T_META_ALL_VALID_ARCHS(YES))
{
	pthread_t thread;
	void *join_result = NULL;
	T_ASSERT_POSIX_ZERO(pthread_create(&thread, NULL, run, NULL), NULL);
	T_ASSERT_POSIX_ZERO(pthread_cancel(thread), NULL);
	T_ASSERT_POSIX_ZERO(pthread_join(thread, &join_result), NULL);
	T_ASSERT_EQ(join_result, PTHREAD_CANCELED, NULL);
}
