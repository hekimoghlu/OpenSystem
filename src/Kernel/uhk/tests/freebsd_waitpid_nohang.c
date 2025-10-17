/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 22, 2024.
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
#include <sys/cdefs.h>
#include <sys/wait.h>

#include <darwintest.h>
#include <signal.h>
#include <unistd.h>

T_GLOBAL_META(T_META_RUN_CONCURRENTLY(true));

T_DECL(waitpid_nohang, "FreeBSDarwin--waitpid_nohang", T_META_TAG_VM_PREFERRED)
{
	pid_t child, pid;
	int status, r;
	siginfo_t siginfo;

	child = fork();
	T_ASSERT_POSIX_SUCCESS(child, "child forked successfully");
	if (child == 0) {
		sleep(10);
		_exit(1);
	}

	status = 42;
	pid = waitpid(child, &status, WNOHANG);
	T_ASSERT_POSIX_ZERO(pid, "waitpid call is successful");
	T_EXPECT_EQ(status, 42, "status is unaffected as expected");

	r = kill(child, SIGTERM);
	T_ASSERT_POSIX_ZERO(r, "signal sent successfully");
	r = waitid(P_PID, (id_t)child, &siginfo, WEXITED | WNOWAIT);
	T_ASSERT_POSIX_SUCCESS(r, "waitid call successful");

	status = -1;
	pid = waitpid(child, &status, WNOHANG);
	T_ASSERT_EQ(pid, child, "waitpid returns correct pid");
	T_EXPECT_EQ(WIFSIGNALED(status), true, "child was signaled");
	T_EXPECT_EQ(WTERMSIG(status), SIGTERM, "child was sent SIGTERM");
}
