/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 25, 2022.
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
 * return minutes offset from absolute timezone expression
 *
 *	[[-+]hh[:mm[:ss]]]
 *	[-+]hhmm
 *
 * if e is non-null then it points to the first unrecognized char in s
 * d returned if no offset in s
 */

int
tmgoff(register const char* s, char** e, int d)
{
	register int	n = d;
	int		east;
	const char*	t = s;

	if ((east = *s == '+') || *s == '-')
	{
		s++;
		if (isdigit(*s) && isdigit(*(s + 1)))
		{
			n = ((*s - '0') * 10 + (*(s + 1) - '0')) * 60;
			s += 2;
			if (*s == ':')
				s++;
			if (isdigit(*s) && isdigit(*(s + 1)))
			{
				n += ((*s - '0') * 10 + (*(s + 1) - '0'));
				s += 2;
				if (*s == ':')
					s++;
				if (isdigit(*s) && isdigit(*(s + 1)))
					s += 2;
			}
			if (east)
				n = -n;
			t = s;
		}
	}
	if (e)
		*e = (char*)t;
	return n;
}
