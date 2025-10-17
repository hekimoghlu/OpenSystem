/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 9, 2024.
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
#include <string.h>
#include <unistd.h>
#include <os/assumes.h>

#include "darwintest_defaults.h"


static uintptr_t stackaddr;
static const size_t stacksize = 4096 * 8;

static void *function(void *arg) {
	// Use the stack...
	char buffer[BUFSIZ];
	strlcpy(buffer, arg, sizeof(buffer));
	strlcat(buffer, arg, sizeof(buffer));

	T_ASSERT_LT((uintptr_t)__builtin_frame_address(0), stackaddr, NULL);
	T_ASSERT_GT((uintptr_t)__builtin_frame_address(0), stackaddr - stacksize, NULL);

	return (void *)(uintptr_t)strlen(buffer);
}

T_DECL(custom_stack, "creating a pthread with a custom stack",
		T_META_ALL_VALID_ARCHS(YES)){
	char *arg = "This is a test and only a test of the pthread stackaddr system.\n";
	stackaddr = (uintptr_t)valloc(stacksize);
	stackaddr += stacksize; // address starts at top of stack.

	pthread_t thread;
	pthread_attr_t attr;

	T_ASSERT_POSIX_ZERO(pthread_attr_init(&attr), NULL);
	T_ASSERT_POSIX_ZERO(pthread_attr_setstacksize(&attr, stacksize), NULL);
	T_ASSERT_POSIX_ZERO(pthread_attr_setstackaddr(&attr, (void *)stackaddr), NULL);

	T_ASSERT_POSIX_ZERO(pthread_create(&thread, &attr, function, arg), NULL);

	void *result;
	T_ASSERT_POSIX_ZERO(pthread_join(thread, &result), NULL);
	T_ASSERT_EQ((uintptr_t)result, (uintptr_t)strlen(arg)*2, "thread should return correct value");

	free((void*)(stackaddr - stacksize));
}
