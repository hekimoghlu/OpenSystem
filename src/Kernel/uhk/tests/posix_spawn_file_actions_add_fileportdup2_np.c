/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 13, 2022.
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

#include <errno.h>
#include <libproc.h>
#include <signal.h>
#include <spawn.h>
#include <spawn_private.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/proc_info.h>
#include <sys/spawn_internal.h>
#include <sys/sysctl.h>
#include <sysexits.h>
#include <unistd.h>
#include <sys/fileport.h>

T_GLOBAL_META(T_META_RUN_CONCURRENTLY(true));

T_DECL(posix_spawn_file_actions_add_fileportdup2_np,
    "Check posix_spawnattr for posix_spawn_file_actions_add_fileportdup2_np",
    T_META_ASROOT(true), T_META_TAG_VM_PREFERRED)
{
	posix_spawnattr_t attr;
	posix_spawn_file_actions_t fact;
	int ret, pipes[2];
	mach_port_t mp;

	ret = pipe(pipes);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(ret, "pipe");

	ret = fileport_makeport(pipes[1], &mp);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(ret, "fileport_makefd");

	ret = posix_spawnattr_init(&attr);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(ret, "posix_spawnattr_init");

	ret = posix_spawn_file_actions_init(&fact);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(ret, "posix_spawn_file_actions_init");

	ret = posix_spawn_file_actions_add_fileportdup2_np(&fact, mp, STDOUT_FILENO);
	T_ASSERT_POSIX_SUCCESS(ret, "posix_spawn_file_actions_add_fileportdup2_np");

	char * const prog = "/bin/echo";
	char * const argv_child[] = { prog, "1", NULL };
	pid_t child_pid;
	extern char   **environ;

	ret = posix_spawn(&child_pid, prog, &fact, &attr, argv_child, environ);
	T_ASSERT_POSIX_SUCCESS(ret, "posix_spawn");

	ret = posix_spawn_file_actions_destroy(&fact);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(ret, "posix_spawn_file_actions_destroy");

	ret = posix_spawnattr_destroy(&attr);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(ret, "posix_spawnattr_destroy");

	T_LOG("parent: spawned child with pid %d\n", child_pid);

	int status = 0;
	int waitpid_result = waitpid(child_pid, &status, 0);
	T_ASSERT_POSIX_SUCCESS(waitpid_result, "waitpid");
	T_ASSERT_EQ(waitpid_result, child_pid, "waitpid should return child we spawned");
	T_ASSERT_EQ(WIFEXITED(status), 1, "child should have exited normally");
	T_ASSERT_EQ(WEXITSTATUS(status), EX_OK, "child should have exited with success");

	char buf[1];
	ssize_t rc = read(pipes[0], buf, sizeof(buf));
	T_ASSERT_POSIX_SUCCESS(rc, "read");
	T_ASSERT_EQ(rc, 1l, "should have read one byte");
	T_ASSERT_EQ(buf[0], '1', "should have read '1'");
}
