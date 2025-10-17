/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 14, 2023.
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

#include <darwin_shim.h>
#include <darwintest.h>
#include <darwintest_utils.h>
#include <perfdata/perfdata.h>

#include <spawn.h>
#include <sys/wait.h>

#define DTRACE_PATH "/usr/sbin/dtrace"
#define DTRACE_SCRIPT "BEGIN { exit(0) }"
#define ITERATIONS 8

static hrtime_t
run_dtrace(void)
{
	char *args[] = {DTRACE_PATH, "-n", DTRACE_SCRIPT, NULL};
	int status, err;
	pid_t pid;
	hrtime_t begin = gethrtime();
	err = posix_spawn(&pid, args[0], NULL, NULL, args, NULL); \
	if (err) {
		T_FAIL("failed to spawn %s", args[0]);
	}
	waitpid(pid, &status, 0);
	if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
		T_FAIL("%s didn't exit properly", args[0]);
	}

	return gethrtime() - begin;
}

T_DECL(dtrace_launchtime, "measure the time to launch dtrace", T_META_CHECK_LEAKS(false), T_META_BOOTARGS_SET("-unsafe_kernel_text"))
{
	char filename[MAXPATHLEN] = "dtrace.launchtime." PD_FILE_EXT;
	dt_resultfile(filename, sizeof(filename));
	T_LOG("perfdata file: %s\n", filename);
	pdwriter_t wr = pdwriter_open(filename, "dtrace.launchtime", 1, 0);
	T_WITH_ERRNO;
	T_ASSERT_NOTNULL(wr, "pdwriter_open %s", filename);

	for (int i = 0; i < ITERATIONS; i++) {
		hrtime_t time = run_dtrace();
		pdwriter_new_value(wr, "launch_time", pdunit_nanoseconds, time);
	}

	pdwriter_close(wr);
}
