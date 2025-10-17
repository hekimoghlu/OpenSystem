/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 2, 2024.
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
 * in place replace of first occurrence of /match/ with /replace/ in path
 * end of path returned
 */

#define _AST_API_H	1

#include <ast.h>

char*
pathrepl(char* path, const char* match, const char* replace)
{
	return pathrepl_20100601(path, PATH_MAX, match, replace);
}

#undef	_AST_API_H

#include <ast_api.h>

char*
pathrepl_20100601(register char* path, size_t size, const char* match, register const char* replace)
{
	register const char*	m = match;
	register const char*	r;
	char*			t;

	if (!match)
		match = "";
	if (!replace)
		replace = "";
	if (streq(match, replace))
		return(path + strlen(path));
	if (!size)
		size = strlen(path) + 1;
	for (;;)
	{
		while (*path && *path++ != '/');
		if (!*path) break;
		if (*path == *m)
		{
			t = path;
			while (*m && *m++ == *path) path++;
			if (!*m && *path == '/')
			{
				register char*	p;

				p = t;
				r = replace;
				while (p < path && *r) *p++ = *r++;
				if (p < path) while (*p++ = *path++);
				else if (*r && p >= path)
				{
					register char*	u;

					t = path + strlen(path);
					u = t + strlen(r);
					while (t >= path) *u-- = *t--;
					while (*r) *p++ = *r++;
				}
				else p += strlen(p) + 1;
				return(p - 1);
			}
			path = t;
			m = match;
		}
	}
	return(path);
}
