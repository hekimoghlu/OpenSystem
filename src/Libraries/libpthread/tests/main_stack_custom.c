/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 11, 2021.
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

#include <stdlib.h>
#include <pthread.h>
#include "darwintest_defaults.h"
#include <machine/vmparam.h>

T_DECL(main_stack_custom, "tests the reported values for a custom main thread stack"){
	T_EXPECT_EQ((size_t)STACKSIZE, pthread_get_stacksize_np(pthread_self()), NULL);

	const uintptr_t stackaddr = (uintptr_t)pthread_get_stackaddr_np(pthread_self());
	size_t stacksize = pthread_get_stacksize_np(pthread_self());
	T_LOG("stack: %zx -> %zx (+%zx)", stackaddr - stacksize, stackaddr, stacksize);
	T_EXPECT_LT((uintptr_t)__builtin_frame_address(0), stackaddr, NULL);
	T_EXPECT_GT((uintptr_t)__builtin_frame_address(0), stackaddr - stacksize, NULL);

	struct rlimit lim;
	T_QUIET; T_ASSERT_POSIX_SUCCESS(getrlimit(RLIMIT_STACK, &lim), NULL);
	lim.rlim_cur = lim.rlim_cur + 32 * PAGE_SIZE;
	T_EXPECT_EQ(setrlimit(RLIMIT_STACK, &lim), -1, "setrlimit for stack should fail with custom stack");
	T_EXPECT_EQ((size_t)STACKSIZE, pthread_get_stacksize_np(pthread_self()), "reported stacksize shouldn't change");
}
