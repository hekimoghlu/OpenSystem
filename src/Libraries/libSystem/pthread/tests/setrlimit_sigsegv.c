/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 21, 2025.
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

#include "darwintest_defaults.h"
#include <spawn.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/resource.h>


extern char **environ;

T_DECL(setrlimit_overflow_segfault,
		"sigsegv is sent when stack(limit set with setrlimit) is fully used.",
		T_META_IGNORECRASHES(".*stackoverflow_crash.*"),
		T_META_CHECK_LEAKS(NO),
		T_META_ALL_VALID_ARCHS(YES),
		T_META_ASROOT(YES))
{
	pid_t child_pid = 0;
	int rv = 0;

	struct rlimit lim, save;
	T_ASSERT_POSIX_SUCCESS(getrlimit(RLIMIT_STACK, &save), NULL);
	T_ASSERT_POSIX_SUCCESS(getrlimit(RLIMIT_STACK, &lim), NULL);
	T_LOG("parent: stack limits cur=%llx max=%llx", lim.rlim_cur, lim.rlim_max);
	lim.rlim_cur = lim.rlim_cur/8;
	T_ASSERT_POSIX_SUCCESS(setrlimit(RLIMIT_STACK, &lim), NULL);
	int status = 0;
	int exit_signal = 0;

	char *crash_cmd[] = { "./assets/stackoverflow_crash", NULL };
	posix_spawn_file_actions_t fact;
	posix_spawn_file_actions_init(&fact);
	T_ASSERT_POSIX_SUCCESS(posix_spawn_file_actions_addinherit_np(&fact, STDIN_FILENO), NULL);
	T_ASSERT_POSIX_SUCCESS(posix_spawn_file_actions_addinherit_np(&fact, STDOUT_FILENO), NULL);
	T_ASSERT_POSIX_SUCCESS(posix_spawn_file_actions_addinherit_np(&fact, STDERR_FILENO), NULL);
	T_LOG("spawning %s", crash_cmd[0]);
	rv = posix_spawn(&child_pid, crash_cmd[0], &fact, NULL, crash_cmd, environ);
	T_ASSERT_POSIX_SUCCESS(rv, "spawning the stackoverflow program");

	T_LOG("parent: waiting for child process with pid %d", child_pid);
	wait(&status);
	T_LOG("parent: child process exited. status=%d", WEXITSTATUS(status));


	T_ASSERT_POSIX_SUCCESS(setrlimit(RLIMIT_STACK, &save), "Restore original limtis");
	posix_spawn_file_actions_destroy(&fact);

	T_ASSERT_TRUE(WIFSIGNALED(status), "child exit with a signal");
	exit_signal = WTERMSIG(status);
	T_ASSERT_EQ(exit_signal, SIGSEGV, "child should receive SIGSEGV");

	return;
}
