/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 28, 2024.
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
//	systemx.c
//	copyfile_test
//	Stolen from the test routines from the apfs project.
//
#include <fcntl.h>
#include <libgen.h>
#include <spawn.h>
#include <stdarg.h>
#include <stdbool.h>
#include <unistd.h>
#include <sys/stat.h>

#include "systemx.h"
#include "test_utils.h"


int __attribute__((sentinel)) systemx(const char *prog, ...)
{
	const char *args[64];

	va_list ap;

	const char **parg = args;

	*parg++ = basename((char *)prog);

	va_start(ap, prog);

	bool quiet = false, quiet_stderr = false;

	while ((*parg = va_arg(ap, char *))) {
		if (*parg == SYSTEMX_QUIET) {
			quiet = true;
		} else if (*parg == SYSTEMX_QUIET_STDERR)
			quiet_stderr = true;
		else
			++parg;
	}

	va_end(ap);

	posix_spawn_file_actions_t facts, *pfacts = NULL;

	if (quiet || quiet_stderr) {
		posix_spawn_file_actions_init(&facts);
		if (quiet)
			posix_spawn_file_actions_addopen(&facts, STDOUT_FILENO, "/dev/null", O_APPEND, 0);
		if (quiet_stderr)
			posix_spawn_file_actions_addopen(&facts, STDERR_FILENO, "/dev/null", O_APPEND, 0);
		pfacts = &facts;
	}

	pid_t pid;
	assert_no_err(errno = posix_spawn(&pid, prog, pfacts, NULL,
									  (char * const *)args, NULL));

	if (pfacts)
		posix_spawn_file_actions_destroy(pfacts);

	int status;
	assert(ignore_eintr(waitpid(pid, &status, 0), -1) == pid);

	if (WIFEXITED(status))
		return WEXITSTATUS(status);
	else
		return -1;
}
