/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 23, 2022.
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
#include <sys/spawn_internal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/kauth.h>
#include <sys/proc_info.h>
#include <sys/sysctl.h>
#include <sysexits.h>
#include <unistd.h>
#include <kern/cs_blobs.h>
#include <sys/codesign.h>

T_GLOBAL_META(T_META_RUN_CONCURRENTLY(true),
    T_META_RADAR_COMPONENT_NAME("xnu"),
    T_META_RADAR_COMPONENT_VERSION("codesigning"));

T_DECL(cs_launch_type_t, "Check posix_spawnattr for launch type",
    T_META_ASROOT(true))
{
	posix_spawnattr_t attr;
	int ret;

	ret = posix_spawnattr_init(&attr);
	T_QUIET;
	T_ASSERT_POSIX_ZERO(ret, "posix_spawnattr_init");

	ret = posix_spawnattr_set_launch_type_np(&attr, CS_LAUNCH_TYPE_SYSTEM_SERVICE);
	T_QUIET;
	T_ASSERT_POSIX_ZERO(ret, "posix_spawnattr_set_launch_type_np");


	char * const    prog = "/bin/ls";
	char * const    argv_child[] = { prog,
		                         NULL, };
	pid_t           child_pid;
	extern char   **environ;

	ret = posix_spawn(&child_pid, prog, NULL, &attr, argv_child, environ);
	T_ASSERT_POSIX_ZERO(ret, "posix_spawn");

	T_LOG("parent: spawned child with pid %d\n", child_pid);

	ret = posix_spawnattr_destroy(&attr);
	T_QUIET;
	T_ASSERT_POSIX_ZERO(ret, "posix_spawnattr_destroy");

	T_LOG("parent: waiting for child process");

	int status = 0;
	int waitpid_result = waitpid(child_pid, &status, 0);
	T_ASSERT_EQ(waitpid_result, child_pid, "waitpid should return child we spawned");
	T_ASSERT_EQ(WIFEXITED(status), 1, "child should have exited normally");
	T_ASSERT_EQ(WEXITSTATUS(status), EX_OK, "child should have exited with success");
}
