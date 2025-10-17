/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 29, 2024.
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
 * escape optget() special chars in s and write to sp
 * esc == '?' or ':' also escaped
 */

#include <optlib.h>
#include <ctype.h>

int
optesc(Sfio_t* sp, register const char* s, int esc)
{
	register const char*	m;
	register int		c;

	if (*s == '[' && *(s + 1) == '+' && *(s + 2) == '?')
	{
		c = strlen(s);
		if (s[c - 1] == ']')
		{
			sfprintf(sp, "%-.*s", c - 4, s + 3);
			return 0;
		}
	}
	if (esc != '?' && esc != ':')
		esc = 0;
	while (c = *s++)
	{
		if (isalnum(c))
		{
			for (m = s - 1; isalnum(*s); s++);
			if (isalpha(c) && *s == '(' && isdigit(*(s + 1)) && *(s + 2) == ')')
			{
				sfputc(sp, '\b');
				sfwrite(sp, m, s - m);
				sfputc(sp, '\b');
				sfwrite(sp, s, 3);
				s += 3;
			}
			else
				sfwrite(sp, m, s - m);
		}
		else if (c == '-' && *s == '-' || c == '<')
		{
			m = s - 1;
			if (c == '-')
				s++;
			else if (*s == '/')
				s++;
			while (isalnum(*s))
				s++;
			if (c == '<' && *s == '>' || isspace(*s) || *s == 0 || *s == '=' || *s == ':' || *s == ';' || *s == '.' || *s == ',')
			{
				sfputc(sp, '\b');
				sfwrite(sp, m, s - m);
				sfputc(sp, '\b');
			}
			else
				sfwrite(sp, m, s - m);
		}
		else
		{
			if (c == ']' || c == esc)
				sfputc(sp, c);
			sfputc(sp, c);
		}
	}
	return 0;
}
