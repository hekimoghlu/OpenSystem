/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 21, 2023.
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
 * single dir support for pathaccess()
 */

#define _AST_API_H	1

#include <ast.h>

/*
 * building 3d flirts with the dark side
 */

#if _BLD_3d

#undef	pathcat
#define pathcat_20100601	_3d_pathcat

#else

char*
pathcat(char* path, const char* dirs, int sep, const char* a, const char* b)
{
	return pathcat_20100601(dirs, sep, a, b, path, PATH_MAX);
}

#endif

#undef	_AST_API

#include <ast_api.h>

char*
pathcat_20100601(register const char* dirs, int sep, const char* a, register const char* b, char* path, size_t size)
{
	register char*	s;
	register char*	e;

	s = path;
	e = path + size;
	while (*dirs && *dirs != sep)
	{
		if (s >= e)
			return 0;
		*s++ = *dirs++;
	}
	if (s != path)
	{
		if (s >= e)
			return 0;
		*s++ = '/';
	}
	if (a)
	{
		while (*s = *a++)
			if (++s >= e)
				return 0;
		if (b)
		{
			if (s >= e)
				return 0;
			*s++ = '/';
		}
	}
	else if (!b)
		b = ".";
	if (b)
		do
		{
			if (s >= e)
				return 0;
		} while (*s++ = *b++);
	return *dirs ? (char*)++dirs : 0;
}
