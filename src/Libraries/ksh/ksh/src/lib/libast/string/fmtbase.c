/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 1, 2023.
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
 * return base b representation for n
 * if p!=0 then base prefix is included
 * otherwise if n==0 or b==0 then output is signed base 10
 */

#include <ast.h>

char*
fmtbase(intmax_t n, int b, int p)
{
	char*	buf;
	int	z;

	if (!p)
	{
		if (!n)
			return "0";
		if (!b)
			return fmtint(n, 0);
		if (b == 10)
			return fmtint(n, 1);
	}
	buf = fmtbuf(z = 72);
	sfsprintf(buf, z, p ? "%#..*I*u" : "%..*I*u", b, sizeof(n), n);
	return buf;
}

#if __OBSOLETE__ < 20140101

#undef	fmtbasell

#if defined(__EXPORT__)
#define extern		__EXPORT__
#endif

extern char*
fmtbasell(intmax_t n, int b, int p)
{
	return fmtbase(n, b, p);
}

#undef	extern

#endif
