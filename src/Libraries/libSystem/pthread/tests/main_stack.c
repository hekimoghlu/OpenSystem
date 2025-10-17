/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 19, 2022.
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

T_DECL(main_stack, "tests the reported values for the main thread stack",
		T_META_CHECK_LEAKS(NO), T_META_ALL_VALID_ARCHS(YES)){
	const uintptr_t stackaddr = (uintptr_t)pthread_get_stackaddr_np(pthread_self());
	const size_t stacksize = pthread_get_stacksize_np(pthread_self());
	T_LOG("stack: %zx -> %zx (+%zx)", stackaddr - stacksize, stackaddr, stacksize);
	T_EXPECT_LT((uintptr_t)__builtin_frame_address(0), stackaddr, NULL);
	T_EXPECT_GT((uintptr_t)__builtin_frame_address(0), stackaddr - stacksize, NULL);

	struct rlimit lim;
	T_ASSERT_POSIX_SUCCESS(getrlimit(RLIMIT_STACK, &lim), NULL);
	T_EXPECT_EQ((size_t)lim.rlim_cur, pthread_get_stacksize_np(pthread_self()), "reported rlimit should match stacksize");

	lim.rlim_cur = lim.rlim_cur / 8;
	T_ASSERT_POSIX_SUCCESS(setrlimit(RLIMIT_STACK, &lim), NULL);

	T_EXPECTFAIL;
	T_EXPECT_EQ((size_t)lim.rlim_cur, pthread_get_stacksize_np(pthread_self()), "new rlimit should should match stacksize");
}
