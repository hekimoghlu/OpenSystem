/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 13, 2022.
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
 * regex collation symbol support
 */

#include "reglib.h"

/*
 * return the collating symbol delimited by [c c], where c is either '=' or '.'
 * s points to the first char after the initial [
 * if e!=0 it is set to point to the next char in s on return
 *
 * the collating symbol is converted to multibyte in <buf,size>
 * the return value is:
 *	-1	syntax error / invalid collating element
 *	>=0	size with 0-terminated mb character (*wc != 0)
 *		or collating element (*wc == 0) in buf
 */

int
regcollate(register const char* s, char** e, char* buf, size_t size, wchar_t* wc)
{
	register int			c;
	register char*			b;
	register char*			x;
	const char*			t;
	int				i;
	int				r;
	int				term;
	wchar_t				w;
	char				xfm[256];
	char				tmp[sizeof(xfm)];

	if (size < 2 || (term = *s) != '.' && term != '=' || !*++s || *s == term && *(s + 1) == ']')
		goto nope;
	t = s;
	w = mbchar(s);
	if ((r = (s - t)) > 1)
	{
		if (*s++ != term || *s++ != ']')
			goto oops;
		goto done;
	}
	if (*s == term && *(s + 1) == ']')
	{
		s += 2;
		goto done;
	}
	b = buf;
	x = buf + size - 2;
	s = t;
	for (;;)
	{
		if (!(c = *s++))
			goto oops;
		if (c == term)
		{
			if (!(c = *s++))
				goto oops;
			if (c != term)
			{
				if (c != ']')
					goto oops;
				break;
			}
		}
		if (b < x)
			*b++ = c;
	}
	r = s - t - 2;
	w = 0;
	if (b >= x)
		goto done;
	*b = 0;
	for (i = 0; i < r && i < sizeof(tmp) - 1; i++)
		tmp[i] = '0';
	tmp[i] = 0;
	if (mbxfrm(xfm, buf, sizeof(xfm)) >= mbxfrm(xfm, tmp, sizeof(xfm)))
		goto nope;
	t = (const char*)buf;
 done:
	if (r <= size && (char*)t != buf)
	{
		memcpy(buf, t, r);
		if (r < size)
			buf[r] = 0;
	}
	if (wc)
		*wc = w;
	if (e)
		*e = (char*)s;
	return r;
 oops:
 	s--;
 nope:
	if (e)
		*e = (char*)s;
	return -1;
}
