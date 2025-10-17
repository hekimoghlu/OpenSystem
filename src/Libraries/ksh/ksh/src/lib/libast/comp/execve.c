/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 5, 2022.
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
#pragma prototyped

#include <ast.h>

#if _lib_execve

NoN(execve)

#else

#include <sig.h>
#include <wait.h>
#include <error.h>

static pid_t		childpid;

static void
execsig(int sig)
{
	kill(childpid, sig);
	signal(sig, execsig);
}

#if defined(__EXPORT__)
#define extern	__EXPORT__
#endif

extern int
execve(const char* path, char* const argv[], char* const arge[])
{
	int	status;

	if ((childpid = spawnve(path, argv, arge)) < 0)
		return(-1);
	for (status = 0; status < 64; status++)
		signal(status, execsig);
	while (waitpid(childpid, &status, 0) == -1)
		if (errno != EINTR) return(-1);
	if (WIFSIGNALED(status))
	{
		signal(WTERMSIG(status), SIG_DFL);
		kill(getpid(), WTERMSIG(status));
		pause();
	}
	else status = WEXITSTATUS(status);
	exit(status);
}

#endif
