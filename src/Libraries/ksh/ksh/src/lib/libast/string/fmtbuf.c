/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 5, 2022.
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

#include <ast.h>

/*
 * return small format buffer chunk of size n
 * spin lock for thread access
 * format buffers are short lived
 * only one concurrent buffer with size > sizeof(buf)
 */

static char		buf[16 * 1024];
static char*		nxt = buf;
static int		lck = -1;

static char*		big;
static size_t		bigsiz;

char*
fmtbuf(size_t n)
{
	register char*	cur;

	while (++lck)
		lck--;
	if (n > (&buf[elementsof(buf)] - nxt))
	{
		if (n > elementsof(buf))
		{
			if (n > bigsiz)
			{
				bigsiz = roundof(n, 8 * 1024);
				if (!(big = newof(big, char, bigsiz, 0)))
				{
					lck--;
					return 0;
				}
			}
			lck--;
			return big;
		}
		nxt = buf;
	}
	cur = nxt;
	nxt += n;
	lck--;
	return cur;
}
