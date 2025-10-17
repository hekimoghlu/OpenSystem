/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 29, 2023.
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
 * AT&T Bell Laboratories
 * force current command to run under dir preroot
 */

#include <ast.h>
#include <preroot.h>

#if FS_PREROOT

#include <option.h>

void
setpreroot(register char** argv, const char* dir)
{
	register char*	s;
	register char**	ap;
	int		argc;
	char*		cmd;
	char**		av;
	char		buf[PATH_MAX];

	if ((argv || (argv = opt_info.argv)) && (dir || (dir = getenv(PR_BASE)) && *dir) && !ispreroot(dir) && (*(cmd = *argv++) == '/' || (cmd = pathpath(cmd, NiL, PATH_ABSOLUTE|PATH_REGULAR|PATH_EXECUTE, buf, sizeof(buf)))))
	{
		argc = 3;
		for (ap = argv; *ap++; argc++);
		if (av = newof(0, char*, argc, 0))
		{
			ap = av;
			*ap++ = PR_COMMAND;
			*ap++ = (char*)dir;
			*ap++ = cmd;
			while (*ap++ = *argv++);
			if (!(s = getenv(PR_SILENT)) || !*s)
			{
				sfprintf(sfstderr, "+");
				ap = av;
				while (s = *ap++)
					sfprintf(sfstderr, " %s", s);
				sfprintf(sfstderr, "\n");
				sfsync(sfstderr);
			}
			execv(*av, av);
			free(av);
		}
	}
}

#else

NoN(setpreroot)

#endif
