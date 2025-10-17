/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 30, 2023.
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
#include <darwintest_multiprocess.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <pthread.h>
#include <stdlib.h>
#include <signal.h>

static void
signal_handler(int sig, siginfo_t *sip __unused, void *ucontext __unused)
{
	if (sig == SIGPIPE) {
		T_FAIL("Received SIGPIPE");
	}

	exit(141);
}

static void *
thread_read(void *arg)
{
	int fd = (int) (uintptr_t)arg;
	char buf[10];

	read(fd, buf, 10);
	T_LOG("thread returned from read");
	return 0;
}

T_DECL(pipe_drain,
    "test a pipe with multiple read descriptor could close one descriptor and drain that descriptor")
{
	int pipe_fd[2];
	int dup_fd;
	int ret;
	char buf[10] = "Hello";
	pthread_t thread;

	/* Install the signal handler for SIGPIPE */

	struct sigaction sa = {
		.sa_sigaction = signal_handler,
		.sa_flags = SA_SIGINFO
	};
	sigfillset(&sa.sa_mask);

	T_QUIET; T_ASSERT_POSIX_ZERO(sigaction(SIGPIPE, &sa, NULL), NULL);

	ret = pipe(pipe_fd);
	T_EXPECT_EQ(ret, 0, NULL);

	dup_fd = dup(pipe_fd[0]);
	T_EXPECT_GE(dup_fd, 0, NULL);

	pthread_create(&thread, NULL, thread_read, (void *) (uintptr_t) pipe_fd[0]);

	sleep(5);

	close(pipe_fd[0]);
	ret = (int)write(pipe_fd[1], buf, strlen(buf) + 1);
	T_EXPECT_EQ(ret, (int)strlen(buf) + 1, NULL);
}
