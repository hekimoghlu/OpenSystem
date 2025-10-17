/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 13, 2025.
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
#include <stdio.h>
#include <stdlib.h>
#include "darwintest_defaults.h"

#if defined(__arm64__)
#define PTHREAD_T_OFFSET (12*1024)
#else
#define PTHREAD_T_OFFSET (0)
#endif

static void *
function(void *arg)
{
	size_t expected_size = (size_t)(uintptr_t)arg;
	T_ASSERT_EQ(pthread_get_stacksize_np(pthread_self()), expected_size,
			"saw expected pthread_get_stacksize_np");
	return NULL;
}

T_DECL(stack_size_default, "stack size of default pthread",
		T_META_ALL_VALID_ARCHS(YES))
{
	static const size_t dflsize = 512 * 1024;
	pthread_t thread;
	pthread_attr_t attr;

	T_ASSERT_POSIX_ZERO(pthread_attr_init(&attr), NULL);
	T_ASSERT_POSIX_ZERO(pthread_create(&thread, &attr, function,
			(void *)(dflsize + PTHREAD_T_OFFSET)), NULL);
	T_ASSERT_POSIX_ZERO(pthread_join(thread, NULL), NULL);
}

T_DECL(stack_size_customsize, "stack size of thread with custom stack size",
		T_META_ALL_VALID_ARCHS(YES))
{
	static const size_t stksize = 768 * 1024;
	pthread_t thread;
	pthread_attr_t attr;

	T_ASSERT_POSIX_ZERO(pthread_attr_init(&attr), NULL);
	T_ASSERT_POSIX_ZERO(pthread_attr_setstacksize(&attr, stksize), NULL);
	T_ASSERT_POSIX_ZERO(pthread_create(&thread, &attr, function,
			(void *)(stksize + PTHREAD_T_OFFSET)), NULL);
	T_ASSERT_POSIX_ZERO(pthread_join(thread, NULL), NULL);
}

T_DECL(stack_size_customaddr, "stack size of thread with custom stack addr",
		T_META_ALL_VALID_ARCHS(YES))
{
	static const size_t stksize = 512 * 1024;
	pthread_t thread;
	pthread_attr_t attr;

	uintptr_t stackaddr = (uintptr_t)valloc(stksize);
	stackaddr += stksize; // address is top of stack

	T_ASSERT_POSIX_ZERO(pthread_attr_init(&attr), NULL);
	T_ASSERT_POSIX_ZERO(pthread_attr_setstackaddr(&attr, (void *)stackaddr),
			NULL);
	T_ASSERT_POSIX_ZERO(pthread_create(&thread, &attr, function,
			(void *)stksize), NULL);
	T_ASSERT_POSIX_ZERO(pthread_join(thread, NULL), NULL);
	free((void *)(stackaddr - stksize));
}

T_DECL(stack_size_custom, "stack size of thread with custom stack addr+size",
		T_META_ALL_VALID_ARCHS(YES))
{
	static const size_t stksize = 768 * 1024;
	pthread_t thread;
	pthread_attr_t attr;

	uintptr_t stackaddr = (uintptr_t)valloc(stksize);
	stackaddr += stksize; // address is top of stack

	T_ASSERT_POSIX_ZERO(pthread_attr_init(&attr), NULL);
	T_ASSERT_POSIX_ZERO(pthread_attr_setstackaddr(&attr, (void *)stackaddr),
			NULL);
	T_ASSERT_POSIX_ZERO(pthread_attr_setstacksize(&attr, stksize), NULL);
	T_ASSERT_POSIX_ZERO(pthread_create(&thread, &attr, function,
			(void *)stksize), NULL);
	T_ASSERT_POSIX_ZERO(pthread_join(thread, NULL), NULL);
	free((void *)(stackaddr - stksize));
}
