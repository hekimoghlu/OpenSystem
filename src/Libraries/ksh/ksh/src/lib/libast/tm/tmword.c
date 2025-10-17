/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 7, 2022.
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
 * time conversion support
 */

#include <ast.h>
#include <tm.h>
#include <ctype.h>

/*
 * match s against t ignoring case and .'s
 *
 * suf is an n element table of suffixes that may trail s
 * if all isalpha() chars in s match then 1 is returned
 * and if e is non-null it will point to the first unmatched
 * char in s, otherwise 0 is returned
 */

int
tmword(register const char* s, char** e, register const char* t, char** suf, int n)
{
	register int	c;
	const char*	b;

	if (*s && *t)
	{
		b = s;
		while (c = *s++)
		{
			if (c != '.')
			{
				if (!isalpha(c) || c != *t && (islower(c) ? toupper(c) : tolower(c)) != *t)
					break;
				t++;
			}
		}
		s--;
		if (!isalpha(c))
		{
			if (c == '_')
				s++;
			if (e)
				*e = (char*)s;
			return s > b;
		}
		if (!*t && s > (b + 1))
		{
			b = s;
			while (n-- && (t = *suf++))
			{
				s = b;
				while (isalpha(c = *s++) && (c == *t || (islower(c) ? toupper(c) : tolower(c)) == *t)) t++;
				if (!*t && !isalpha(c))
				{
					if (c != '_')
						s--;
					if (e)
						*e = (char*)s;
					return 1;
				}
			}
		}
	}
	return 0;
}
