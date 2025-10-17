/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 21, 2024.
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

/*
 * n is minutes west of UTC
 *
 * append p and SHHMM part of n to s
 * where S is + or -
 *
 * n ignored if n==d
 * end of s is returned
 */

char*
tmpoff(register char* s, size_t z, register const char* p, register int n, int d)
{
	register char*	e = s + z;

	while (s < e && (*s = *p++))
		s++;
	if (n != d && s < e)
	{
		if (n < 0)
		{
			n = -n;
			*s++ = '+';
		}
		else
			*s++ = '-';
		s += sfsprintf(s, e - s, "%02d%s%02d", n / 60, d == -24*60 ? ":" : "", n % 60);
	}
	return s;
}
