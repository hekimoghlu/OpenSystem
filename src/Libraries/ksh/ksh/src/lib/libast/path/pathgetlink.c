/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 11, 2021.
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
*/

#include "univlib.h"

#ifdef UNIV_MAX

#include <ctype.h>

#endif

/*
 * return external representation for symbolic link text of name in buf
 * the link text string length is returned
 */

int
pathgetlink(const char* name, char* buf, int siz)
{
	int	n;

	if ((n = readlink(name, buf, siz)) < 0) return(-1);
	if (n >= siz)
	{
		errno = EINVAL;
		return(-1);
	}
	buf[n] = 0;
#ifdef UNIV_MAX
	if (isspace(*buf))
	{
		register char*	s;
		register char*	t;
		register char*	u;
		register char*	v;
		int		match = 0;
		char		tmp[PATH_MAX];

		s = buf;
		t = tmp;
		while (isalnum(*++s) || *s == '_' || *s == '.');
		if (*s++)
		{
			for (;;)
			{
				if (!*s || isspace(*s))
				{
					if (match)
					{
						*t = 0;
						n = t - tmp;
						strcpy(buf, tmp);
					}
					break;
				}
				if (t >= &tmp[sizeof(tmp)]) break;
				*t++ = *s++;
				if (!match && t < &tmp[sizeof(tmp) - univ_size + 1]) for (n = 0; n < UNIV_MAX; n++)
				{
					if (*(v = s - 1) == *(u = univ_name[n]))
					{
						while (*u && *v++ == *u) u++;
						if (!*u)
						{
							match = 1;
							strcpy(t - 1, univ_cond);
							t += univ_size - 1;
							s = v;
							break;
						}
					}
				}
			}
		}
	}
#endif
	return(n);
}
