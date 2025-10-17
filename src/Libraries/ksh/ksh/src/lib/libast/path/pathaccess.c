/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 13, 2023.
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
 * Glenn Fowler
 * AT&T Bell Laboratories
 *
 * return path to file a/b with access mode using : separated dirs
 * both a and b may be 0
 * if a==".." then relative paths in dirs are ignored
 * if (mode&PATH_REGULAR) then path must not be a directory
 * if (mode&PATH_ABSOLUTE) then path must be rooted
 * path returned in path buffer
 */

#define _AST_API_H	1

#include <ast.h>

char*
pathaccess(char* path, const char* dirs, const char* a, const char* b, int mode)
{
	return pathaccess_20100601(dirs, a, b, mode, path, PATH_MAX);
}

#undef	_AST_API_H

#include <ast_api.h>

char*
pathaccess_20100601(register const char* dirs, const char* a, const char* b, register int mode, register char* path, size_t size)
{
	int		sib = a && a[0] == '.' && a[1] == '.' && a[2] == 0;
	int		sep = ':';
	char		cwd[PATH_MAX];

	do
	{
		dirs = pathcat(dirs, sep, a, b, path, size);
		pathcanon(path, size, 0);
		if ((!sib || *path == '/') && pathexists(path, mode))
		{
			if (*path == '/' || !(mode & PATH_ABSOLUTE))
				return path;
			dirs = getcwd(cwd, sizeof(cwd));
			sep = 0;
		}
	} while (dirs);
	return 0;
}
