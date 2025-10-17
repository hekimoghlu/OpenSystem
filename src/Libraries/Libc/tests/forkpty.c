/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 31, 2023.
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

#include <sys/resource.h>
#include <sys/wait.h>

#include <errno.h>
#include <pwd.h>
#include <stdio.h>
#include <unistd.h>
#include <util.h>

#include <darwintest.h>
#include <TargetConditionals.h>

T_GLOBAL_META(T_META_RUN_CONCURRENTLY(true));

T_DECL(forkpty_forkfail,
    "Check for fd leak when fork() fails",
    T_META_CHECK_LEAKS(false),
    T_META_ENABLED(TARGET_OS_OSX))
{
	struct rlimit orl, nrl;
	struct passwd *pwd;
	pid_t pid;
	int prevfd, fd, pty;

	T_SETUPBEGIN;
	if (geteuid() == 0) {
		/* the setrlimit() trick won't work if we're root */
		T_ASSERT_NOTNULL(pwd = getpwnam("nobody"), NULL);
		T_ASSERT_POSIX_SUCCESS(setgid(pwd->pw_gid), NULL);
		T_ASSERT_POSIX_SUCCESS(setuid(pwd->pw_uid), NULL);
	}
	T_ASSERT_POSIX_SUCCESS(getrlimit(RLIMIT_NPROC, &orl), NULL);
	nrl = orl;
	nrl.rlim_cur = 1;
	T_ASSERT_POSIX_SUCCESS(setrlimit(RLIMIT_NPROC, &nrl), NULL);
	T_ASSERT_POSIX_SUCCESS(fd = dup(0), NULL);
	T_ASSERT_POSIX_SUCCESS(close(fd), NULL);
	T_SETUPEND;
	pid = forkpty(&pty, NULL, NULL, NULL);
	if (pid == 0) {
		/* child - fork() unexpectedly succeeded */
		_exit(0);
	}
	T_EXPECT_POSIX_FAILURE(pid, EAGAIN, "expected fork() to fail");
	if (pid > 0) {
		/* parent - fork() unexpectedly succeeded */
		(void)waitpid(pid, NULL, 0);
	}
	prevfd = fd;
	T_ASSERT_POSIX_SUCCESS(fd = dup(0), NULL);
	T_EXPECT_EQ(fd, prevfd, "expected same fd as previously");
}

