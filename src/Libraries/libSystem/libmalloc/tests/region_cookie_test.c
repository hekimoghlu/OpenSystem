/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 2, 2021.
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

//
//  region_cookie_test.c
//  libsystem_malloc
//
//  Created by Kim Topley on 10/24/18.
//

#include <darwintest.h>
#include <../src/internal.h>

T_GLOBAL_META(T_META_RUN_CONCURRENTLY(true));

T_DECL(tiny_region_cookie_test, "Crash on corruption of tiny region cookie",
		T_META_ENVVAR("MallocNanoZone=0"),
		T_META_ENVVAR("MallocMaxMagazines=1"),
		T_META_IGNORECRASHES("region_cookie_test"))
{
	pid_t child_pid = fork();
	T_ASSERT_NE(child_pid, -1, "Fork failed");

	if (!child_pid) {
		// MallocMaxMagazines=1 ensures these allocations come from the
		// same magazine.
		void *ptr1 = malloc(1);
		void *ptr2 = malloc(1);
		T_ASSERT_NOTNULL(ptr1, "Allocation #1 succeeded");
		T_ASSERT_NOTNULL(ptr2, "Allocation #2 succeeded");

		// Corrupt the region cookie for both pointers (just in case they
		// are in different regions).
		void *region = TINY_REGION_FOR_PTR(ptr1);
		REGION_COOKIE_FOR_TINY_REGION(region)++;

		region = TINY_REGION_FOR_PTR(ptr2);
		REGION_COOKIE_FOR_TINY_REGION(region)++;

		free(ptr1);
		free(ptr2);

		// Should not get here
		T_FAIL("Tiny region cookie corruption test failed");
	} else {
		int status;
		pid_t wait_pid = waitpid(child_pid, &status, 0);
		T_ASSERT_EQ(wait_pid, child_pid, "Got child status");
		T_ASSERT_TRUE(WIFSIGNALED(status), "Child terminated by signal");
		T_ASSERT_EQ(WTERMSIG(status), SIGABRT, "Child aborted");
	}
}

T_DECL(small_region_cookie_test, "Crash on corruption of small region cookie",
		T_META_ENVVAR("MallocNanoZone=0"),
		T_META_ENVVAR("MallocMaxMagazines=1"),
		T_META_IGNORECRASHES("region_cookie_test"))
{
	pid_t child_pid = fork();
	T_ASSERT_NE(child_pid, -1, "Fork failed");

	if (!child_pid) {
		// MallocMaxMagazines=1 ensures these allocations come from the
		// same magazine.
		void *ptr1 = malloc(TINY_LIMIT_THRESHOLD + 1);
		void *ptr2 = malloc(TINY_LIMIT_THRESHOLD + 1);
		T_ASSERT_NOTNULL(ptr1, "Allocation #1 succeeded");
		T_ASSERT_NOTNULL(ptr2, "Allocation #2 succeeded");

		// Corrupt the region cookie for both pointers (just in case they
		// are in different regions).
		void *region = SMALL_REGION_FOR_PTR(ptr1);
		REGION_COOKIE_FOR_SMALL_REGION(region)++;

		region = TINY_REGION_FOR_PTR(ptr2);
		REGION_COOKIE_FOR_SMALL_REGION(region)++;

		free(ptr1);
		free(ptr2);

		// Should not get here
		T_FAIL("Small region cookie corruption test failed");
	} else {
		int status;
		pid_t wait_pid = waitpid(child_pid, &status, 0);
		T_ASSERT_EQ(wait_pid, child_pid, "Got child status");
		T_ASSERT_TRUE(WIFSIGNALED(status), "Child terminated by signal");
		T_ASSERT_EQ(WTERMSIG(status), SIGABRT, "Child aborted");
	}
}

