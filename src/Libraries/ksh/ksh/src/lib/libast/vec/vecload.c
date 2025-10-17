/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 13, 2025.
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
 * string vector load support
 */

#include <ast.h>
#include <vecargs.h>

/*
 * load a string vector from lines in buf
 * buf may be modified on return
 *
 * each line in buf is treated as a new vector element
 * lines with # as first char are comments
 * \ as the last char joins consecutive lines
 *
 * the vector ends with a 0 sentinel
 *
 * the string array pointer is returned
 */

char**
vecload(char* buf)
{
	register char*	s;
	register int	n;
	register char**	p;
	char**		vec;

	vec = 0;
	n = (*buf == '#') ? -1 : 0;
	for (s = buf;; s++)
	{
		if (*s == '\n')
		{
			if (s > buf && *(s - 1) == '\\') *(s - 1) = *s = ' ';
			else
			{
				*s = 0;
				if (*(s + 1) != '#')
				{
					n++;
					if (!*(s + 1)) break;
				}
			}
		}
		else if (!*s)
		{
			n++;
			break;
		}
	}
	if (n < 0) n = 0;
	if (p = newof(0, char*, n + 3, 0))
	{
		*p++ = s = buf;
		vec = ++p;
		if (n > 0) for (;;)
		{
			if (*s != '#')
			{
				*p++ = s;
				if (--n <= 0) break;
			}
			while (*s) s++;
			s++;
		}
		*p = 0;
		*(vec - 1) = (char*)p;
	}
	return(vec);
}
