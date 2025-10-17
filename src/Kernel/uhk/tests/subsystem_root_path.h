/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 16, 2024.
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

#include <unistd.h>
#include <spawn.h>
#include <sys/wait.h>

#define SUBSYSTEM_ROOT_PATH_KEY "subsystem_root_path"

#define HELPER_BEHAVIOR_NOT_SET    "not_set"
#define HELPER_BEHAVIOR_SET        "set"
#define HELPER_BEHAVIOR_FORK_EXEC  "fork_exec"
#define HELPER_BEHAVIOR_SPAWN      "spawn"

static int
_spawn_and_wait(char ** args, posix_spawnattr_t *attr)
{
	int pid;
	int status;

	if (posix_spawn(&pid, args[0], NULL, attr, args, NULL)) {
		return -1;
	}
	if (waitpid(pid, &status, 0) < 0) {
		return -1;
	}

	if (WIFEXITED(status) && (WEXITSTATUS(status) == 0)) {
		return 0;
	}

	return -1;
}
