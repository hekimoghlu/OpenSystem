/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 16, 2023.
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
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#include <sys/codesign.h>
#include <signal.h>

#include <darwintest.h>
#include <darwintest_utils.h>

T_GLOBAL_META(T_META_RUN_CONCURRENTLY(true));

T_DECL(static_binary,
    "Verify that static binaries have CS_NO_UNTRUSTED_HELPERS set", T_META_TAG_VM_PREFERRED) {
	int ret;
	pid_t pid;
	char *launch_argv[] = {"./static_binary", NULL};
	ret = dt_launch_tool(&pid, launch_argv, /*start_suspended*/ true, NULL, NULL);
	T_QUIET;
	T_ASSERT_POSIX_SUCCESS(ret, "dt_launch_tool on static binary");

	uint32_t status = 0;
	ret = csops(pid, CS_OPS_STATUS, &status, sizeof(status));
	T_QUIET;
	T_EXPECT_POSIX_SUCCESS(ret, "request CS_OPS_STATUS on static binary");

	if (!ret) {
		T_EXPECT_BITS_SET(status, CS_NO_UNTRUSTED_HELPERS, "CS_NO_UNTRUSTED_HELPERS should be set on static binary");
	}

	ret = kill(pid, SIGCONT);
	T_QUIET;
	T_ASSERT_POSIX_SUCCESS(ret, "SIGCONT on static binary");

	int exitstatus, signal;
	dt_waitpid(pid, &exitstatus, &signal, 30);
	T_QUIET;
	T_ASSERT_EQ(signal, 0, "static binary exited");
	T_QUIET;
	T_ASSERT_EQ(exitstatus, 42, "static binary exited with code 42");
}
