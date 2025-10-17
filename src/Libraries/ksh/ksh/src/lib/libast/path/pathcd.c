/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 24, 2022.
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
 * K. P. Vo
 * G. S. Fowler
 * AT&T Research
 */

#include <ast.h>
#include <error.h>
#include <stk.h>

#if DEBUG

#undef	PATH_MAX

#define PATH_MAX	16

static int
vchdir(const char* path)
{
	int	n;

	if (strlen(path) >= PATH_MAX)
	{
		errno = ENAMETOOLONG;
		n = -1;
	}
	else n = chdir(path);
	return n;
}

#define chdir(p)	vchdir(p)

#endif

/*
 * set the current directory to path
 * if path is long and home!=0 then pathcd(home,0)
 * is called on intermediate chdir errors
 */

int
pathcd(const char* path, const char* home)
{
	register char*	p = (char*)path;
	register char*	s;
	register int	n;
	int		i;
	int		r;

	r = 0;
	for (;;)
	{
		/*
		 * this should work 99% of the time
		 */

		if (!chdir(p))
			return r;

		/*
		 * chdir failed
		 */

		if ((n = strlen(p)) < PATH_MAX)
			return -1;
#ifdef ENAMETOOLONG
		if (errno != ENAMETOOLONG)
			return -1;
#endif

		/*
		 * path is too long -- copy so it can be modified in place
		 */

		i = stktell(stkstd);
		sfputr(stkstd, p, 0);
		stkseek(stkstd, i);
		p = stkptr(stkstd, i);
		for (;;)
		{
			/*
			 * get a short prefix component
			 */

			s = p + PATH_MAX;
			while (--s >= p && *s != '/');
			if (s <= p)
				break;

			/*
			 * chdir to the prefix
			 */

			*s++ = 0;
			if (chdir(p))
				break;

			/*
			 * do the remainder
			 */

			if ((n -= s - p) < PATH_MAX)
			{
				if (chdir(s))
					break;
				return r;
			}
			p = s;
		}

		/*
		 * try to recover back to home
		 */

		if (!(p = (char*)home))
			return -1;
		home = 0;
		r = -1;
	}
}
