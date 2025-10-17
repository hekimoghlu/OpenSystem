/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 24, 2023.
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
#include <darwintest.h>

#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/wait.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.quicktest"),
	T_META_CHECK_LEAKS(false),
	T_META_RUN_CONCURRENTLY(true)
	);

T_DECL(getpriority_setpriority, "Tests getpriority and setpriority system calls", T_META_ASROOT(true))
{
	int my_priority;
	int my_new_priority;

	/* getpriority returns scheduling priority so -1 is a valid value */
	errno       = 0;
	my_priority = getpriority(PRIO_PROCESS, 0);

	T_WITH_ERRNO;
	T_ASSERT_FALSE(my_priority == -1 && errno != 0, "Verify getpriority is successful", NULL);

	/* change scheduling priority*/
	my_new_priority = (my_priority == PRIO_MIN) ? (my_priority + 10) : (PRIO_MIN);

	T_WITH_ERRNO;
	T_ASSERT_POSIX_SUCCESS(setpriority(PRIO_PROCESS, 0, my_new_priority), "Change scheduling priority", NULL);

	/* verify change */
	errno       = 0;
	my_priority = getpriority(PRIO_PROCESS, 0);
	T_WITH_ERRNO;
	T_ASSERT_FALSE(my_priority == -1 && errno != 0, "Verify getpriority change is successful", NULL);

	T_WITH_ERRNO;
	T_ASSERT_EQ(my_priority, my_new_priority, "Verify setpriority correctly set scheduling priority", NULL);

	/* reset scheduling priority */
	T_WITH_ERRNO;
	T_ASSERT_POSIX_SUCCESS(setpriority(PRIO_PROCESS, 0, 0), "Reset scheduling priority", NULL);
}
