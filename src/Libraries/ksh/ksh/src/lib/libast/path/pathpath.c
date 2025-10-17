/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 23, 2022.
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
 * AT&T Research
 *
 * return full path to p with mode access using $PATH
 * a!=0 enables related root search
 * a!=0 && a!="" searches a dir first
 * the related root must have a bin subdir
 * p==0 sets the cached relative dir to a
 * full path returned in path buffer
 * if path==0 then the space is malloc'd
 */

#define _AST_API_H	1

#include <ast.h>

char*
pathpath(char* path, const char* p, const char* a, int mode)
{
	return pathpath_20100601(p, a, mode, path, PATH_MAX);
}

#undef	_AST_API_H

#include <ast_api.h>

char*
pathpath_20100601(const char* p, const char* a, int mode, register char* path, size_t size)
{
	register char*	s;
	char*		x;
	char		buf[PATH_MAX];

	static char*	cmd;

	if (!path)
	{
		path = buf;
		if (!size || size > sizeof(buf))
			size = sizeof(buf);
	}
	if (!p)
	{
		if (cmd)
			free(cmd);
		cmd = a ? strdup(a) : (char*)0;
		return 0;
	}
	if (strlen(p) < size)
	{
		strcpy(path, p);
		if (pathexists(path, mode))
		{
			if (*p != '/' && (mode & PATH_ABSOLUTE))
			{
				getcwd(buf, sizeof(buf));
				s = buf + strlen(buf);
				sfsprintf(s, sizeof(buf) - (s - buf), "/%s", p);
				if (path != buf)
					strcpy(path, buf);
			}
			return (path == buf) ? strdup(path) : path;
		}
	}
	if (*p == '/')
		a = 0;
	else if (s = (char*)a)
	{
		x = s;
		if (strchr(p, '/'))
		{
			a = p;
			p = "..";
		}
		else
			a = 0;
		if ((!cmd || *cmd) && (strchr(s, '/') || (s = cmd)))
		{
			if (!cmd && *s == '/')
				cmd = strdup(s);
			if (strlen(s) < (sizeof(buf) - 6))
			{
				s = strcopy(path, s);
				for (;;)
				{
					do if (s <= path) goto normal; while (*--s == '/');
					do if (s <= path) goto normal; while (*--s != '/');
					strcpy(s + 1, "bin");
					if (pathexists(path, PATH_EXECUTE))
					{
						if (s = pathaccess(path, p, a, mode, path, size))
							return path == buf ? strdup(s) : s;
						goto normal;
					}
				}
			normal: ;
			}
		}
	}
	x = !a && strchr(p, '/') ? "" : pathbin();
	if (!(s = pathaccess(x, p, a, mode, path, size)) && !*x && (x = getenv("FPATH")))
		s = pathaccess(x, p, a, mode, path, size);
	return (s && path == buf) ? strdup(s) : s;
}
