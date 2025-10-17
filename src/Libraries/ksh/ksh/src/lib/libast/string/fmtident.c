/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 4, 2021.
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

#define IDENT	01
#define USAGE	02

/*
 * format what(1) and/or ident(1) string a
 */

char*
fmtident(const char* a)
{
	register char*	s = (char*)a;
	register char*	t;
	char*		buf;
	int		i;

	i = 0;
	for (;;)
	{
		while (isspace(*s))
			s++;
		if (s[0] == '[')
		{
			while (*++s && *s != '\n');
			i |= USAGE;
		}
		else if (s[0] == '@' && s[1] == '(' && s[2] == '#' && s[3] == ')')
			s += 4;
		else if (s[0] == '$' && s[1] == 'I' && s[2] == 'd' && s[3] == ':' && isspace(s[4]))
		{
			s += 5;
			i |= IDENT;
		}
		else
			break;
	}
	if (i)
	{
		i &= IDENT;
		for (t = s; isprint(*t) && *t != '\n'; t++)
			if (i && t[0] == ' ' && t[1] == '$')
				break;
		while (t > s && isspace(t[-1]))
			t--;
		i = t - s;
		buf = fmtbuf(i + 1);
		memcpy(buf, s, i);
		s = buf;
		s[i] = 0;
	}
	return s;
}
