/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 5, 2022.
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
 * return RE expression given strmatch() pattern
 * 0 returned for invalid RE
 */

#include <ast.h>

typedef struct Stack_s
{
	char*		beg;
	short		len;
	short		min;
} Stack_t;

char*
fmtre(const char* as)
{
	register char*		s = (char*)as;
	register int		c;
	register char*		t;
	register Stack_t*	p;
	char*			x;
	int			n;
	int			end;
	char*			buf;
	Stack_t			stack[32];

	end = 1;
	c = 2 * strlen(s) + 1;
	t = buf = fmtbuf(c);
	p = stack;
	if (*s != '*' || *(s + 1) == '(' || *(s + 1) == '-' && *(s + 2) == '(')
		*t++ = '^';
	else
		s++;
	for (;;)
	{
		switch (c = *s++)
		{
		case 0:
			break;
		case '\\':
			if (!(c = *s++) || c == '{' || c == '}')
				return 0;
			*t++ = '\\';
			if ((*t++ = c) == '(' && *s == '|')
			{
				*t++ = *s++;
				goto logical;
			}
			continue;
		case '[':
			*t++ = c;
			n = 0;
			if ((c = *s++) == '!')
			{
				*t++ = '^';
				c = *s++;
			}
			else if (c == '^')
			{
				if ((c = *s++) == ']')
				{
					*(t - 1) = '\\';
					*t++ = '^';
					continue;
				}
				n = '^';
			}
			for (;;)
			{
				if (!(*t++ = c))
					return 0;
				if ((c = *s++) == ']')
				{
					if (n)
						*t++ = n;
					*t++ = c;
					break;
				}
			}
			continue;
		case '{':
			for (x = s; *x && *x != '}'; x++);
			if (*x++ && (*x == '(' || *x == '-' && *(x + 1) == '('))
			{
				if (p >= &stack[elementsof(stack)])
					return 0;
				p->beg = s - 1;
				s = x;
				p->len = s - p->beg;
				if (p->min = *s == '-')
					s++;
				p++;
				*t++ = *s++;
			}
			else
				*t++ = c;
			continue;
		case '*':
			if (!*s)
			{
				end = 0;
				break;
			}
			/*FALLTHROUGH*/
		case '?':
		case '+':
		case '@':
		case '!':
		case '~':
			if (*s == '(' || c != '~' && *s == '-' && *(s + 1) == '(')
			{
				if (p >= &stack[elementsof(stack)])
					return 0;
				p->beg = s - 1;
				if (c == '~')
				{
					if (*(s + 1) == 'E' && *(s + 2) == ')')
					{
						for (s += 3; *t = *s; t++, s++);
						continue;
					}
					p->len = 0;
					p->min = 0;
					*t++ = *s++;
					*t++ = '?';
				}
				else
				{
					p->len = c != '@';
					if (p->min = *s == '-')
						s++;
					*t++ = *s++;
				}
				p++;
			}
			else
			{
				switch (c)
				{
				case '*':
					*t++ = '.';
					break;
				case '?':
					c = '.';
					break;
				case '+':
				case '!':
					*t++ = '\\';
					break;
				}
				*t++ = c;
			}
			continue;
		case '(':
			if (p >= &stack[elementsof(stack)])
				return 0;
			p->beg = s - 1;
			p->len = 0;
			p->min = 0;
			p++;
			*t++ = c;
			continue;
		case ')':
			if (p == stack)
				return 0;
			*t++ = c;
			p--;
			for (c = 0; c < p->len; c++)
				*t++ = p->beg[c];
			if (p->min)
				*t++ = '?';
			continue;
		case '^':
		case '.':
		case '$':
			*t++ = '\\';
			*t++ = c;
			continue;
		case '|':
			if (t == buf || *(t - 1) == '(')
				return 0;
		logical:
			if (!*s || *s == ')')
				return 0;
			/*FALLTHROUGH*/
		default:
			*t++ = c;
			continue;
		}
		break;
	}
	if (p != stack)
		return 0;
	if (end)
		*t++ = '$';
	*t = 0;
	return buf;
}
