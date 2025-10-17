/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 29, 2021.
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
#include <ctype.h>

/*
 * version strncmp(3)
 */

int
strnvcmp(register const char* a, register const char* b, size_t n)
{
	register const char*	ae;
	register const char*	be;
	register unsigned long	na;
	register unsigned long	nb;

	ae = a + n;
	be = b + n;
	for (;;)
	{
		if (a >= ae)
		{
			if (b >= be)
				return 0;
			return 1;
		}
		else if (b >= be)
			return -1;
		if (isdigit(*a) && isdigit(*b))
		{
			na = nb = 0;
			while (a < ae && isdigit(*a))
				na = na * 10 + *a++ - '0';
			while (b < be && isdigit(*b))
				nb = nb * 10 + *b++ - '0';
			if (na < nb)
				return -1;
			if (na > nb)
				return 1;
		}
		else if (*a != *b)
			break;
		else if (!*a)
			return 0;
		else
		{
			a++;
			b++;
		}
	}
	if (*a == 0)
		return -1;
	if (*b == 0)
		return 1;
	if (*a == '.')
		return -1;
	if (*b == '.')
		return 1;
	if (*a == '-')
		return -1;
	if (*b == '-')
		return 1;
	return *a < *b ? -1 : 1;
}
