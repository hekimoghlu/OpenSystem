/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 20, 2023.
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
#include "leaks.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

int
leaks(int argc, char *const *argv)
{
	int result = 1;
	pid_t child;
	pid_t parent = getpid();

	child = fork();
	switch (child)
	{
	case -1:
		/* Fork failed we're hosed. */
		fprintf(stderr, "fork: %s", strerror(errno));
		break;
	case 0:
	{
		/* child. */
		char **argvec = (char **)malloc((argc + 2) * sizeof(char *));
		char pidstr[8];
		int ix;
	
		snprintf(pidstr, sizeof(pidstr), "%d", parent);
		argvec[0] = "/usr/bin/leaks";
		for (ix = 1; ix < argc; ++ix)
			argvec[ix] = argv[ix];
		argvec[ix] = pidstr;
		argvec[ix + 1] = NULL;

		execv(argvec[0], argvec);
		fprintf(stderr, "exec: %s", strerror(errno));
		_exit(1);
	}
	default:
	{
		/* Parent. */
		int status = 0;
		for (;;)
		{
			/* Wait for the child to exit. */
			pid_t waited_pid = waitpid(child, &status, 0);
			if (waited_pid == -1)
			{
				int error = errno;
				/* Keep going if we get interupted but bail out on any
				   other error. */
				if (error == EINTR)
					continue;

				fprintf(stderr, "waitpid %d: %s", status, strerror(errno));
				break;
			}

			if (WIFEXITED(status))
			{
				if (WEXITSTATUS(status))
				{
					/* Force usage message. */
					result = 2;
					fprintf(stderr, "leaks exited: %d", result);
				}
				break;
			}
			else if (WIFSIGNALED(status))
			{
				fprintf(stderr, "leaks terminated by signal: %d", WTERMSIG(status));
				break;
			}
		}
		break;
	}
	}

	return result;
}
