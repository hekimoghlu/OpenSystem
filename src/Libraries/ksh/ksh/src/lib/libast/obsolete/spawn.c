/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 20, 2023.
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

/*
 * OBSOLETE 20030321 -- use spawnveg()
 */

#include <ast_lib.h>

#if !_lib_spawnve
#define spawnve		______spawnve
#endif
#if !_lib_spawnvpe
#define spawnvpe	______spawnvpe
#endif
#if !_lib_spawnvp
#define spawnvp		______spawnvp
#endif
#if !_lib_spawnlp
#define spawnlp		______spawnlp
#endif

#include <ast.h>
#include <error.h>

#if !_lib_spawnve
#undef	spawnve
#endif
#if !_lib_spawnvpe
#undef	spawnvpe
#endif
#if !_lib_spawnvp
#undef	spawnvp
#endif
#if !_lib_spawnlp
#undef	spawnlp
#endif

#if defined(__EXPORT__)
#define extern	__EXPORT__
#endif

#if _lib_spawnve

NoN(spawnve)

#else

extern pid_t
spawnve(const char* cmd, char* const argv[], char* const envv[])
{
	return spawnveg(cmd, argv, envv, 0);
}

#endif

#if _lib_spawnvpe

NoN(spawnvpe)

#else

extern pid_t
spawnvpe(const char* name, char* const argv[], char* const envv[])
{
	register const char*	path = name;
	pid_t			pid;
	char			buffer[PATH_MAX];

	if (*path != '/')
		path = pathpath(name, NULL, PATH_REGULAR|PATH_EXECUTE, buffer, sizeof(buffer));
	if ((pid = spawnve(path, argv, envv)) >= 0)
		return pid;
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
			pid = spawnve(path, newargv, environ);
			free(newargv);
		}
		else
			errno = ENOMEM;
	}
	return pid;
}

#endif

#if _lib_spawnvp

NoN(spawnvp)

#else

extern pid_t
spawnvp(const char* name, char* const argv[])
{
	return spawnvpe(name, argv, environ);
}

#endif

#if _lib_spawnlp

NoN(spawnlp)

#else

extern pid_t
spawnlp(const char* name, const char* arg, ...)
{
	va_list	ap;
	pid_t	pid;

	va_start(ap, arg);
	pid = spawnvp(name, (char* const*)&arg);
	va_end(ap);
	return pid;
}

#endif
