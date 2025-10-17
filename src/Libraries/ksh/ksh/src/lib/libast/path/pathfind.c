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
 * include style search support
 */

#include <ast.h>
#include <error.h>
#include <ls.h>

#define directory(p,s)	(stat((p),(s))>=0&&S_ISDIR((s)->st_mode))
#define regular(p,s)	(stat((p),(s))>=0&&(S_ISREG((s)->st_mode)||streq(p,"/dev/null")))

typedef struct Dir_s			/* directory list element	*/
{
	struct Dir_s*	next;		/* next in list			*/
	char		dir[1];		/* directory path		*/
} Dir_t;

static struct				/* directory list state		*/
{
	Dir_t*		head;		/* directory list head		*/
	Dir_t*		tail;		/* directory list tail		*/
} state;

/*
 * append dir to pathfind() include list
 */

int
pathinclude(const char* dir)
{
	register Dir_t*	dp;
	struct stat	st;

	if (dir && *dir && !streq(dir, ".") && directory(dir, &st))
	{
		for (dp = state.head; dp; dp = dp->next)
			if (streq(dir, dp->dir))
				return 0;
		if (!(dp = oldof(0, Dir_t, 1, strlen(dir))))
			return -1;
		strcpy(dp->dir, dir);
		dp->next = 0;
		if (state.tail)
			state.tail = state.tail->next = dp;
		else
			state.head = state.tail = dp;
	}
	return 0;
}

/*
 * return path to name using pathinclude() list
 * path placed in <buf,size>
 * if lib!=0 then pathpath() attempted after include search
 * if type!=0 and name has no '.' then file.type also attempted
 * any *: prefix in lib is ignored (discipline library dictionary support)
 */

char*
pathfind(const char* name, const char* lib, const char* type, char* buf, size_t size)
{
	register Dir_t*		dp;
	register char*		s;
	char			tmp[PATH_MAX];
	struct stat		st;

	if (((s = strrchr(name, '/')) || (s = (char*)name)) && strchr(s, '.'))
		type = 0;

	/*
	 * always check the unadorned path first
	 * this handles . and absolute paths
	 */

	if (regular(name, &st))
	{
		strncopy(buf, name, size);
		return buf;
	}
	if (type)
	{
		sfsprintf(buf, size, "%s.%s", name, type);
		if (regular(buf, &st))
			return buf;
	}
	if (*name == '/')
		return 0;

	/*
	 * check the directory of the including file
	 * on the assumption that error_info.file is properly stacked
	 */

	if (error_info.file && (s = strrchr(error_info.file, '/')))
	{
		sfsprintf(buf, size, "%-.*s%s", s - error_info.file + 1, error_info.file, name);
		if (regular(buf, &st))
			return buf;
		if (type)
		{
			sfsprintf(buf, size, "%-.*s%s%.s", s - error_info.file + 1, error_info.file, name, type);
			if (regular(buf, &st))
				return buf;
		}
	}

	/*
	 * check the include dir list
	 */

	for (dp = state.head; dp; dp = dp->next)
	{
		sfsprintf(tmp, sizeof(tmp), "%s/%s", dp->dir, name);
		if (pathpath(tmp, "", PATH_REGULAR, buf, size))
			return buf;
		if (type)
		{
			sfsprintf(tmp, sizeof(tmp), "%s/%s.%s", dp->dir, name, type);
			if (pathpath(tmp, "", PATH_REGULAR, buf, size))
				return buf;
		}
	}

	/*
	 * finally a lib related search on PATH
	 */

	if (lib)
	{
		if (s = strrchr((char*)lib, ':'))
			lib = (const char*)s + 1;
		sfsprintf(tmp, sizeof(tmp), "lib/%s/%s", lib, name);
		if (pathpath(tmp, "", PATH_REGULAR, buf, size))
			return buf;
		if (type)
		{
			sfsprintf(tmp, sizeof(tmp), "lib/%s/%s.%s", lib, name, type);
			if (pathpath(tmp, "", PATH_REGULAR, buf, size))
				return buf;
		}
	}
	return 0;
}
