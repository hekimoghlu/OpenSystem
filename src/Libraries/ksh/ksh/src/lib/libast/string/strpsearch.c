/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 10, 2024.
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
 */

#include <ast.h>
#include <ccode.h>
#include <ctype.h>

#if CC_NATIVE == CC_ASCII
#define MAP(m,c)	(c)
#else
#define MAP(m,c)	m[c]
#endif

/*
 * return a pointer to the isalpha() identifier matching
 * name in the CC_ASCII sorted tab of num elements of
 * size siz where the first member of each
 * element is a char*
 *
 * [xxx] brackets optional identifier characters
 * * starts optional identifier characters
 *
 * 0 returned if name not found
 * otherwise if next!=0 then it points to the next
 * unmatched char in name
 */

void*
strpsearch(const void* tab, size_t num, size_t siz, const char* name, char** next)
{
	register char*		lo = (char*)tab;
	register char*		hi = lo + (num - 1) * siz;
	register char*		mid;
#if CC_NATIVE != CC_ASCII
	register unsigned char*	m;
#endif
	register unsigned char*	s;
	register unsigned char*	t;
	register int		c;
	register int		v;
	int			sequential = 0;

#if CC_NATIVE != CC_ASCII
	m = ccmap(CC_NATIVE, CC_ASCII);
#endif
	c = MAP(m, *((unsigned char*)name));
	while (lo <= hi)
	{
		mid = lo + (sequential ? 0 : (((hi - lo) / siz) / 2) * siz);
		if (!(v = c - MAP(m, *(s = *((unsigned char**)mid)))) || *s == '[' && !(v = c - MAP(m, *++s)) && (v = 1))
		{
			t = (unsigned char*)name;
			for (;;)
			{
				if (!v && (*s == '[' || *s == '*'))
				{
					v = 1;
					s++;
				}
				else if (v && *s == ']')
				{
					v = 0;
					s++;
				}
				else if (!isalpha(*t))
				{
					if (v || !*s)
					{
						if (next)
							*next = (char*)t;
						return (void*)mid;
					}
					if (!sequential)
					{
						while ((mid -= siz) >= lo && (s = *((unsigned char**)mid)) && ((c == MAP(m, *s)) || *s == '[' && c == MAP(m, *(s + 1))));
						sequential = 1;
					}
					v = 1;
					break;
				}
				else if (*t != *s)
				{
					v = MAP(m, *t) - MAP(m, *s);
					break;
				}
				else
				{
					t++;
					s++;
				}
			}
		}
		else if (sequential)
			break;
		if (v > 0)
			lo = mid + siz;
		else
			hi = mid - siz;
	}
	return 0;
}
