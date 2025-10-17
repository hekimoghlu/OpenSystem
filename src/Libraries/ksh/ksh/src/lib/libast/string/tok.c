/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 19, 2024.
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
 * token stream routines
 */

#include <ast.h>
#include <tok.h>

#define FLG_RESTORE	01		/* restore string on close	*/
#define FLG_NEWLINE	02		/* return newline token next	*/

typedef struct Tok_s			/* token stream state		*/
{
	union
	{
	char*		end;		/* end ('\0') of last token	*/
	struct Tok_s*	nxt;		/* next in free list		*/
	}		ptr;
	char		chr;		/* replace *end with this	*/
	char		flg;		/* FLG_*			*/
} Tok_t;

static Tok_t*		freelist;

/*
 * open a new token stream on s
 * if f==0 then string is not restored
 */

char*
tokopen(register char* s, int f)
{
	register Tok_t*	p;

	if (p = freelist)
		freelist = freelist->ptr.nxt;
	else if (!(p = newof(0, Tok_t, 1, 0)))
		return 0;
	p->chr = *(p->ptr.end = s);
	p->flg = f ? FLG_RESTORE : 0;
	return (char*)p;
}

/*
 * close a token stream
 * restore the string to its original state
 */

void
tokclose(char* u)
{
	register Tok_t*	p = (Tok_t*)u;

	if (p->flg == FLG_RESTORE && *p->ptr.end != p->chr)
		*p->ptr.end = p->chr;
	p->ptr.nxt = freelist;
	freelist = p;
}

/*
 * return next space separated token
 * "\n" is returned as a token
 * 0 returned when no tokens remain
 * "..." and '...' quotes are honored with \ escapes
 */

char*
tokread(char* u)
{
	register Tok_t*	p = (Tok_t*)u;
	register char*	s;
	register char*	r;
	register int	q;
	register int	c;

	/*
	 * restore string on each call
	 */

	if (!p->chr)
		return 0;
	s = p->ptr.end;
	switch (p->flg)
	{
	case FLG_NEWLINE:
		p->flg = 0;
		return "\n";
	case FLG_RESTORE:
		if (*s != p->chr)
			*s = p->chr;
		break;
	default:
		if (!*s)
			s++;
		break;
	}

	/*
	 * skip leading space
	 */

	while (*s == ' ' || *s == '\t')
		s++;
	if (!*s)
	{
		p->ptr.end = s;
		p->chr = 0;
		return 0;
	}

	/*
	 * find the end of this token
	 */

	r = s;
	q = 0;
	for (;;)
		switch (c = *r++)
		{
		case '\n':
			if (!q)
			{
				if (s == (r - 1))
				{
					if (!p->flg)
					{
						p->ptr.end = r;
						return "\n";
					}
					r++;
				}
				else if (!p->flg)
					p->flg = FLG_NEWLINE;
			}
			/*FALLTHROUGH*/
		case ' ':
		case '\t':
			if (q)
				break;
			/*FALLTHROUGH*/
		case 0:
			if (s == --r)
			{
				p->ptr.end = r;
				p->chr = 0;
			}
			else
			{
				p->chr = *(p->ptr.end = r);
				if (*r)
					*r = 0;
			}
			return s;
		case '\\':
			if (*r)
				r++;
			break;
		case '"':
		case '\'':
			if (c == q)
				q = 0;
			else if (!q)
				q = c;
			break;
		}
}
