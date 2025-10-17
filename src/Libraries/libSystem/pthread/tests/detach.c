/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 9, 2025.
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
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <dispatch/dispatch.h>
#include <sys/mman.h>

#include "darwintest_defaults.h"

static void*
thread_routine(void *loc)
{
	uintptr_t i = (uintptr_t)loc;

	switch (i % 3) {
	case 0:
		usleep(1000);
		break;
	case 1:
		pthread_exit(pthread_self());
		__builtin_unreachable();
	case 2:
		break;
	}
	return NULL;
}

T_DECL(pthread_detach, "Test creating and detaching threads in a loop",
		T_META_CHECK_LEAKS(NO), T_META_ALL_VALID_ARCHS(YES))
{
	const size_t count = 32;
	pthread_t ths[count];

	for (size_t i = 0; i < 100; i++) {
		for (size_t j = 0; j < count; j++) {
			T_ASSERT_POSIX_ZERO(pthread_create(&ths[j], NULL,
							thread_routine, (void *)j), "thread creation");
			T_ASSERT_POSIX_ZERO(pthread_detach(ths[j]), "thread detach");
		}
		usleep(50000);
	}
}
