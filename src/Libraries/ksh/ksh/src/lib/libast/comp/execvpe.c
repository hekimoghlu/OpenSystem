/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 6, 2025.
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

#include <ast_lib.h>

#if _lib_execvpe

#include <ast.h>

NoN(execvpe)

#else

#if defined(__EXPORT__)
__EXPORT__ int execvpe(const char*, char* const[], char* const[]);
#endif

#include <ast.h>
#include <errno.h>

#if defined(__EXPORT__)
#define extern	__EXPORT__
#endif

extern int
execvpe(const char* name, char* const argv[], char* const envv[])
{
	register const char*	path = name;
	char			buffer[PATH_MAX];

	if (*path != '/' && !(path = pathpath(name, NULL, PATH_REGULAR|PATH_EXECUTE, buffer, sizeof(buffer))))
		path = name;
	execve(path, argv, envv);
	if (errno == ENOEXEC)
	{
		register char**	newargv;
		register char**	ov;
		register char**	nv;

		for (ov = (char**)argv; *ov++;);
		if (newargv = newof(0, char*, ov + 1 - (char**)argv, 0))
		{
			nv = newargv;
			*nv++ = "sh";
			*nv++ = (char*)path;
			ov = (char**)argv;
			while (*nv++ = *++ov);
			path = pathshell();
			execve(path, newargv, envv);
			free(newargv);
		}
		else
			errno = ENOMEM;
	}
	return -1;
}

#endif
