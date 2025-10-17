/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 26, 2024.
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
#include <unistd.h>
#include <stdio.h>

#include <darwintest.h>
#include <darwintest_utils.h>

T_GLOBAL_META(
/*
 * We're going to corrupt shared library text, so don't
 * run with other tests.
 */
	T_META_RUN_CONCURRENTLY(false),
	T_META_REQUIRES_SYSCTL_NE("kern.page_protection_type", 2)
	);

/*
 * No system(3c) on watchOS, so provide our own.
 * returns -1 if fails to run
 * returns 0 if process exits normally.
 * returns +n if process exits due to signal N
 */
static int
my_system(const char *command)
{
	pid_t pid;
	int status = 0;
	int signal = 0;
	int err;
	const char *argv[] = {
		"/bin/sh",
		"-c",
		command,
		NULL
	};

	if (dt_launch_tool(&pid, (char **)(void *)argv, FALSE, NULL, NULL)) {
		return -1;
	}

	err = dt_waitpid(pid, &status, &signal, 30);
	if (err) {
		return 0;
	}

	return signal;
}


/*
 * The tests are run in the following order:
 *
 * - call foo
 * - corrupt foo, then call foo
 * - call foo
 *
 * - call atan
 * - corrupt atan, then call atan
 * - call atan
 *
 * The first and last of each should exit normally. The middle one should exit with SIGILL.
 *
 * atan() was picked as a shared region function that isn't likely used by any normal daemons.
 */
T_DECL(text_corruption_recovery, "test detection/recovery of text corruption",
    T_META_IGNORECRASHES(".*text_corruption_helper.*"),
    T_META_ASROOT(true))
{
	int ret;

	ret = my_system("./text_corruption_helper foo");
	T_QUIET; T_ASSERT_EQ(ret, 0, "First call of foo");

	ret = my_system("./text_corruption_helper Xfoo");
	T_QUIET; T_ASSERT_EQ(ret, SIGILL, "Call of corrupted foo");

	ret = my_system("./text_corruption_helper foo");
	T_QUIET; T_ASSERT_EQ(ret, 0, "Fixed call of foo");

	ret = my_system("./text_corruption_helper atan");
	T_QUIET; T_ASSERT_EQ(ret, 0, "First call of atan");

	ret = my_system("./text_corruption_helper Xatan");
	T_QUIET; T_ASSERT_EQ(ret, SIGILL, "Call of corrupted atan");

	ret = my_system("./text_corruption_helper atan");
	T_QUIET; T_ASSERT_EQ(ret, 0, "Fixed call of atan");
}
