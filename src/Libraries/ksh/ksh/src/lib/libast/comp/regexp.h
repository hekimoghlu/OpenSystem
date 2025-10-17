/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 21, 2023.
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
 * regexp interface and partial implementation
 * what a novel approach
 * don't do it again
 *
 * OBSOLETE: use <regex.h>
 */

#ifndef _REGEXP_H
#define _REGEXP_H

#define NBRA		9

typedef struct
{
	char*		re_braslist[NBRA];
	char*		re_braelist[NBRA];
	char*		re_loc1;
	char*		re_loc2;
	char*		re_locs;
	int		re_circf;
	int		re_nbra;
	int		re_nodelim;
	int		re_sed;
} regexp_t;

#define braslist	_re_info.re_braslist
#define braelist	_re_info.re_braelist
#define circf		_re_info.re_circf
#define loc1		_re_info.re_loc1
#define loc2		_re_info.re_loc2
#define locs		_re_info.re_locs
#define nbra		_re_info.re_nbra
#define nodelim		_re_info.re_nodelim
#define sed		_re_info.re_sed

#define advance(a,b)		_re_exec(&_re_info,a,b,1)
#define compile(a,b,c,d)	_re_read(&_re_info,a,b,c,d)
#define step(a,b)		_re_exec(&_re_info,a,b,0)

#if _BLD_ast && defined(__EXPORT__)
#define extern		__EXPORT__
#endif

extern int	_re_comp(regexp_t*, const char*, char*, unsigned int);
extern int	_re_exec(regexp_t*, const char*, const char*, int);
extern char*	_re_putc(int);
extern char*	_re_read(regexp_t*, const char*, char*, const char*, int);

#undef	extern

#ifndef _REGEXP_DECLARE

regexp_t	_re_info;

char*
_re_read(register regexp_t* re, const char* instring, char* ep, const char* endbuf, int seof)
{
	register int		c;

	static const char*	prev;

#ifdef INIT
	INIT;
#endif

	re->re_nodelim = 0;
	if ((c = GETC()) == seof || c == '\n' || c == -1 || c == 0)
	{
		if (c != seof)
		{
			UNGETC(c);
			re->re_nodelim = 1;
		}
		if (!re->re_sed && !prev)
			{ ERROR(41); }
		RETURN((char*)endbuf);
	}
	UNGETC(c);
	prev = 0;
	for (;;)
	{
		if ((c = GETC()) == seof || c == '\n' || c == -1 || c == 0)
		{
			if (re->re_sed)
				{ ERROR(36); }
			UNGETC(c);
			re->re_nodelim = 1;
			break;
		}
		if (c == '\\')
		{
			_re_putc(c);
			if ((c = GETC()) == seof || c == '\n' || c == -1 || c == 0)
				{ ERROR(36); }
		}
		_re_putc(c);
	}
	if (c = _re_comp(re, _re_putc(0), ep, (char*)endbuf - ep))
		{ ERROR(c); }
	prev = endbuf;
	RETURN((char*)prev);
}

#endif

#endif
