/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 23, 2022.
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
 * return 1 if path exisis
 * maintains a cache to minimize stat(2) calls
 * path is modified in-place but restored on return
 * path components checked in pairs to cut stat()'s
 * in half by checking ENOTDIR vs. ENOENT
 * case ignorance infection unavoidable here
 */

#include "lclib.h"

#include <ls.h>
#include <error.h>

typedef struct Tree_s
{
	struct Tree_s*	next;
	struct Tree_s*	tree;
	int		mode;
	char		name[1];
} Tree_t;

int
pathexists(char* path, int mode)
{
	register char*		s;
	register char*		e;
	register Tree_t*	p;
	register Tree_t*	t;
	register int		c;
	char*			ee;
	int			cc;
	int			x;
	struct stat		st;
	int			(*cmp)(const char*, const char*);

	static Tree_t		tree;

	t = &tree;
	e = (c = *path) == '/' ? path + 1 : path;
	cmp = strchr(astconf("PATH_ATTRIBUTES", path, NiL), 'c') ? strcasecmp : strcmp;
	if ((ast.locale.set & (AST_LC_debug|AST_LC_find)) == (AST_LC_debug|AST_LC_find))
		sfprintf(sfstderr, "locale test %s\n", path);
	while (c)
	{
		p = t;
		for (s = e; *e && *e != '/'; e++);
		c = *e;
		*e = 0;
		for (t = p->tree; t && (*cmp)(s, t->name); t = t->next);
		if (!t)
		{
			if (!(t = newof(0, Tree_t, 1, strlen(s))))
			{
				*e = c;
				return 0;
			}
			strcpy(t->name, s);
			t->next = p->tree;
			p->tree = t;
			if (c)
			{
				*e = c;
				for (s = ee = e + 1; *ee && *ee != '/'; ee++);
				cc = *ee;
				*ee = 0;
			}
			else
				ee = 0;
			if ((ast.locale.set & (AST_LC_debug|AST_LC_find)) == (AST_LC_debug|AST_LC_find))
				sfprintf(sfstderr, "locale stat %s\n", path);
			x = stat(path, &st);
			if (ee)
			{
				e = ee;
				c = cc;
				if (!x || errno == ENOENT)
					t->mode = PATH_READ|PATH_EXECUTE;
				if (!(p = newof(0, Tree_t, 1, strlen(s))))
				{
					*e = c;
					return 0;
				}
				strcpy(p->name, s);
				p->next = t->tree;
				t->tree = p;
				t = p;
			}
			if (x)
			{
				*e = c;
				return 0;
			}
			if (st.st_mode & (S_IRUSR|S_IRGRP|S_IROTH))
				t->mode |= PATH_READ;
			if (st.st_mode & (S_IWUSR|S_IWGRP|S_IWOTH))
				t->mode |= PATH_WRITE;
			if (st.st_mode & (S_IXUSR|S_IXGRP|S_IXOTH))
				t->mode |= PATH_EXECUTE;
			if (!S_ISDIR(st.st_mode))
				t->mode |= PATH_REGULAR;
		}
		*e++ = c;
		if (!t->mode || c && (t->mode & PATH_REGULAR))
			return 0;
	}
	mode &= (PATH_READ|PATH_WRITE|PATH_EXECUTE|PATH_REGULAR);
	return (t->mode & mode) == mode;
}
