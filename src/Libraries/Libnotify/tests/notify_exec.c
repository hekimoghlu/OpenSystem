/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 15, 2024.
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

//
//  notify_exec.c
//  darwintests
//
//  Created by Brycen Wershing on 2/10/20.
//

#include <darwintest.h>
#include <notify.h>
#include <notify_private.h>
#include <dispatch/dispatch.h>
#include <stdlib.h>
#include <unistd.h>

extern char **environ;

T_DECL(notify_exec,
       "notify exec",
       T_META("owner", "Core Darwin Daemons & Tools"),
       T_META("as_root", "true"))
{
	pid_t pid = fork();

	if (pid == 0) {
		printf("Child started up\n");

		char *argv[3];
		argv[0] = "notify_test_helper";
		argv[1] = "Continue";
		argv[2] = NULL;

		execve("/AppleInternal/Tests/Libnotify/notify_test_helper" ,argv , environ);
		printf("execve failed with %d\n", errno);
		abort();
	} else {
		int status;
		T_LOG("Fork returned %d", pid);
		pid = waitpid(pid, &status, 0);
		if (pid == -1) {
			T_FAIL("wait4 failed with %d", errno);
			return;
		}

		if (!WIFEXITED(status)) {
			T_FAIL("Unexpected helper termination");
			return;
		}

		int exitStatus = WEXITSTATUS(status);

		if (exitStatus == 42) {
			T_PASS("Succeeded!");
		} else {
			T_FAIL("Failed with %d", exitStatus);
		}
	}
}
