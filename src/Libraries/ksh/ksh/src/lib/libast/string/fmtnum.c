/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 27, 2023.
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
 * return scaled number n
 * string width is 5 chars or less
 * if m>1 then n divided by m before scaling
 */

#include <ast.h>

char*
fmtnum(register unsigned long n, int m)
{
	register int		i;
	register unsigned long	r;
	char*			buf;
	int			z;

	char			suf[2];

	if (m > 1)
	{
		r = n;
		n /= m;
		r -= n;
	}
	else
		r = 0;
	suf[1] = 0;
	if (n < 1024)
		suf[0] = 0;
	else if (n < 1024 * 1024)
	{
		suf[0] = 'k';
		r = ((n % 1024) * 100) / 1024;
		n /= 1024;
	}
	else if (n < 1024 * 1024 * 1024)
	{
		suf[0] = 'm';
		r = ((n % (1024 * 1024)) * 100) / (1024 * 1024);
		n /= 1024 * 1024;
	}
	else
	{
		suf[0] = 'g';
		r = ((n % (1024 * 1024 * 1024)) * 100) / (1024 * 1024 * 1024);
		n /= 1024 * 1024 * 1024;
	}
	if (r)
	{
		if (n >= 100)
			r = 0;
		else if (n >= 10)
		{
			i = 1;
			if (r >= 10)
				r /= 10;
		}
		else
			i = 2;
	}
	buf = fmtbuf(z = 8);
	if (r)
		sfsprintf(buf, z, "%lu.%0*lu%s", n, i, r, suf);
	else
		sfsprintf(buf, z, "%lu%s", n, suf);
	return buf;
}
