/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 21, 2022.
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
 * G. S. Fowler
 * D. G. Korn
 * AT&T Bell Laboratories
 *
 * shell library support
 */

#include <ast.h>
#include <sys/stat.h>

/*
 * return pointer to the full path name of the shell
 *
 * SHELL is read from the environment and must start with /
 *
 * if set-uid or set-gid then the executable and its containing
 * directory must not be owned by the real user/group
 *
 * root/administrator has its own test
 *
 * astconf("SH",NiL,NiL) is returned by default
 *
 * NOTE: csh is rejected because the bsh/csh differentiation is
 *       not done for `csh script arg ...'
 */

char*
pathshell(void)
{
	register char*	sh;
	int		ru;
	int		eu;
	int		rg;
	int		eg;
	struct stat	st;

	static char*	val;

	if ((sh = getenv("SHELL")) && *sh == '/' && strmatch(sh, "*/(sh|*[!cC]sh)*([[:digit:]])?(-+([.[:alnum:]]))?(.exe)"))
	{
		if (!(ru = getuid()) || !eaccess("/bin", W_OK))
		{
			if (stat(sh, &st))
				goto defshell;
			if (ru != st.st_uid && !strmatch(sh, "?(/usr)?(/local)/?([ls])bin/?([[:lower:]])sh?(.exe)"))
				goto defshell;
		}
		else
		{
			eu = geteuid();
			rg = getgid();
			eg = getegid();
			if (ru != eu || rg != eg)
			{
				char*	s;
				char	dir[PATH_MAX];

				s = sh;
				for (;;)
				{
					if (stat(s, &st))
						goto defshell;
					if (ru != eu && st.st_uid == ru)
						goto defshell;
					if (rg != eg && st.st_gid == rg)
						goto defshell;
					if (s != sh)
						break;
					if (strlen(s) >= sizeof(dir))
						goto defshell;
					strcpy(dir, s);
					if (!(s = strrchr(dir, '/')))
						break;
					*s = 0;
					s = dir;
				}
			}
		}
		return sh;
	}
 defshell:
	if (!(sh = val))
	{
		if (!*(sh = astconf("SH", NiL, NiL)) || *sh != '/' || eaccess(sh, X_OK) || !(sh = strdup(sh)))
			sh = "/bin/sh";
		val = sh;
	}
	return sh;
}
