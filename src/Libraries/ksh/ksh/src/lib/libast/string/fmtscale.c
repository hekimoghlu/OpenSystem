/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 3, 2023.
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
 * return number n scaled to metric multiples of k { 1000 1024 }
 * return string length is at most 5 chars + terminating nul
 */

#include <ast.h>
#include <lclib.h>

char*
fmtscale(register Sfulong_t n, int k)
{
	register Sfulong_t	m;
	int			r;
	int			z;
	const char*		u;
	char			suf[3];
	char*			s;
	char*			buf;
	Lc_numeric_t*		p = (Lc_numeric_t*)LCINFO(AST_LC_NUMERIC)->data;

	static const char	scale[] = "bkMGTPE";

	u = scale;
	if (n < 1000)
		r = 0;
	else
	{
		m = 0;
		while (n >= k && *(u + 1))
		{
			m = n;
			n /= k;
			u++;
		}
		if ((r = (10 * (m % k) + (k / 2)) / k) > 9)
		{
			r = 0;
			n++;
		}
		if (k == 1024 && n >= 1000)
		{
			n = 1;
			r = 0;
			u++;
		}
	}
	buf = fmtbuf(z = 8);
	s = suf;
	if (u > scale)
	{
		if (k == 1024)
		{
			*s++ = *u == 'k' ? 'K' : *u;
			*s++ = 'i';
		}
		else
			*s++ = *u;
	}
	*s = 0;
	if (n > 0 && n < 10)
		sfsprintf(buf, z, "%I*u%c%d%s", sizeof(n), n, p->decimal >= 0 ? p->decimal : '.', r, suf);
	else
	{
		if (r >= 5)
			n++;
		sfsprintf(buf, z, "%I*u%s", sizeof(n), n, suf);
	}
	return buf;
}
