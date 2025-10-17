/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 30, 2025.
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

#if _lib_rename

NoN(rename)

#else

#include <error.h>
#include <proc.h>

#ifdef EPERM

static int
mvdir(const char* from, const char* to)
{
	char*			argv[4];
	int			oerrno;

	static const char	mvdir[] = "/usr/lib/mv_dir";

	oerrno = errno;
	if (!eaccess(mvdir, X_OK))
	{
		argv[0] = mvdir;
		argv[1] = from;
		argv[2] = to;
		argv[3] = 0;
		if (!procrun(argv[0], argv, 0))
		{
			errno = oerrno;
			return 0;
		}
	}
	errno = EPERM;
	return -1;
}

#endif

int
rename(const char* from, const char* to)
{
	int	oerrno;
	int	ooerrno;

	ooerrno = errno;
	while (link(from, to))
	{
#ifdef EPERM
		if (errno == EPERM)
		{
			errno = ooerrno;
			return mvdir(from, to);
		}
#endif
		oerrno = errno;
		if (unlink(to))
		{
#ifdef EPERM
			if (errno == EPERM)
			{
				errno = ooerrno;
				return mvdir(from, to);
			}
#endif
			errno = oerrno;
			return -1;
		}
	}
	errno = ooerrno;
	return unlink(from);
}

#endif
