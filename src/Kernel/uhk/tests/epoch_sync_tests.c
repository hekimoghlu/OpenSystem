/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 19, 2022.
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
#include <sys/sysctl.h>
#include <sys/wait.h>

#include <darwintest.h>
#include <signal.h>


T_GLOBAL_META(
	T_META_NAMESPACE("xnu.epoch_sync_tests"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("scheduler"),
	T_META_OWNER("mphalan"));

#define TEST_SYSCTL "debug.test.esync_test"
T_DECL(epoch_sync_test, "Test Epoch Sync",
    T_META_REQUIRES_SYSCTL_EQ(TEST_SYSCTL, 0))
{
	int64_t old = 0;
	size_t old_len = sizeof(old);
	int64_t new = 10;
	size_t new_len = sizeof(new);

	int rc = sysctlbyname(TEST_SYSCTL, &old, &old_len, &new, new_len);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(rc, "sysctlbyname(" TEST_SYSCTL ")");
}

#define TIMEOUT 10

static void
timeout(__unused int ignored)
{
	T_ASSERT_FAIL("child didn't exit in time");
}

#define TEST_WAIT_SYSCTL "debug.test.esync_test_wait"
T_DECL(epoch_sync_test_wait, "Test Epoch Sync Wait",
    T_META_REQUIRES_SYSCTL_EQ(TEST_WAIT_SYSCTL, 0))
{
	int64_t old = 0;
	size_t old_len = sizeof(old);
	int64_t new = 10;
	size_t new_len = sizeof(new);

	pid_t pid = fork();
	T_ASSERT_POSIX_SUCCESS(pid, "fork");

	/* Have the child block in an abortable esync_wait call. */
	if (pid == 0) {
		int rc = sysctlbyname(TEST_WAIT_SYSCTL, &old, &old_len, &new, new_len);
		/*
		 * The only way out of this syscall is if the process is killed.
		 * So nothing after this point should run.
		 */
		T_ASSERT_FAIL("Unexpectedly returned from sysctl (%d)", rc);
	}

	/* Give enough time for the child to block in esync_wait. */
	sleep(1);

	/* Kill the child. */
	int ret = kill(pid, SIGKILL);
	T_ASSERT_POSIX_SUCCESS(ret, "killing child");

	/* Wait a maximum of TIMEOUT seconds for the child to exit. */
	T_ASSERT_NE(signal(SIGALRM, timeout), SIG_ERR, NULL);
	T_ASSERT_POSIX_SUCCESS(alarm(TIMEOUT), NULL);

	int status = 0;
	T_ASSERT_POSIX_SUCCESS(waitpid(pid, &status, 0), "waiting for child");

	/* Check that the child was killed. */
	T_ASSERT_TRUE(WIFSIGNALED(status), "exited due to signal");
	T_ASSERT_EQ(WTERMSIG(status), SIGKILL, "killed with SIGKILL");
}
