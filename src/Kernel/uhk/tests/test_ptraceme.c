/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 28, 2023.
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
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/ptrace.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.spawn"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("spawn"),
	T_META_OWNER("jainam_shah"),
	T_META_RUN_CONCURRENTLY(true)
	);

T_DECL(test_ptraceme, "Test that ptraced process is stopped when it execs", T_META_ASROOT(true), T_META_ENABLED(TARGET_OS_OSX), T_META_TAG_VM_PREFERRED)
{
	int ret;

	T_LOG("Parent %d: Calling fork()", getpid());
	pid_t child_pid = fork();
	if (child_pid == -1) {
		T_FAIL("Fork failed with error: %d: %s", errno, strerror(errno));
	} else if (child_pid == 0) {
		/* Child */
		T_LOG("Child %d: Calling ptrace(PT_TRACE_ME, 0, NULL, 0)", getpid());
		ret = ptrace(PT_TRACE_ME, 0, NULL, 0);
		T_EXPECT_POSIX_SUCCESS(ret, "ptrace PT_TRACE_ME");

		T_LOG("Child %d: Calling execl(\"/bin/echo\", ...)", getpid());
		execl("/bin/echo", "echo", "/bin/echo executed - this should not happen before parent has detached!", NULL);
		T_FAIL("execl failed with error: %d: %s", errno, strerror(errno));
	} else {
		/* Parent */
		T_LOG("Parent %d: Calling waitpid(%d, NULL, WUNTRACED)", getpid(), child_pid);
		int child_status = 0;

		ret = waitpid(child_pid, &child_status, WUNTRACED);
		T_EXPECT_EQ(ret, child_pid, "Waitpid returned status for child pid");

		T_EXPECT_TRUE(WIFSTOPPED(child_status),
		    "Parent %d: waitpid() indicates that child %d is now stopped for tracing", getpid(), child_pid);

		T_LOG("Parent %d: Calling ptrace(PT_DETACH, %d, NULL, 0)", getpid(), child_pid);
		ret = ptrace(PT_DETACH, child_pid, NULL, 0);
		T_EXPECT_POSIX_SUCCESS(ret, "ptrace PT_DETACH");

		T_LOG("Parent %d: Calling kill(%d, SIGTERM)", getpid(), child_pid);
		kill(child_pid, SIGTERM);

		T_LOG("Parent %d: Calling wait(NULL)\n", getpid());
		wait(NULL);

		T_END;
	}
}
