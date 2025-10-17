/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 27, 2024.
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
 * NOTE: mbs* and wcs* are provided to avoid link errors only
 */

#include <ast.h>
#include <wchar.h>

#define STUB	1

#if !_lib_mbtowc
#undef	STUB
size_t
mbtowc(wchar_t* t, const char* s, size_t n)
{
	if (t && n > 0)
		*t = *s;
	return 1;
}
#endif

#if !_lib_mbrtowc
#undef	STUB
size_t
mbrtowc(wchar_t* t, const char* s, size_t n, mbstate_t* q)
{
#if _lib_mbtowc
#undef	STUB
	memset(q, 0, sizeof(*q));
	return mbtowc(t, s, n);
#else
	*q = 0;
	if (t && n > 0)
		*t = *s;
	return 1;
#endif
}
#endif

#if !_lib_mbstowcs
#undef	STUB
size_t
mbstowcs(wchar_t* t, const char* s, size_t n)
{
	register wchar_t*	p = t;
	register wchar_t*	e = t + n;
	register unsigned char*	u = (unsigned char*)s;

	if (t)
		while (p < e && (*p++ = *u++));
	else
		while (p++, *u++);
	return p - t;
}
#endif

#if !_lib_wctomb
#undef	STUB
int
wctomb(char* s, wchar_t c)
{
	if (s)
		*s = c;
	return 1;
}
#endif

#if !_lib_wcrtomb
#undef	STUB
size_t
wcrtomb(char* s, wchar_t c, mbstate_t* q)
{
#if _lib_wctomb
#undef	STUB
	memset(q, 0, sizeof(*q));
	return wctomb(s, c);
#else
	if (s)
		*s = c;
	*q = 0;
	return 1;
#endif
}
#endif

#if !_lib_wcslen
#undef	STUB
size_t
wcslen(const wchar_t* s)
{
	register const wchar_t*	p = s;

	while (*p)
		p++;
	return p - s;
}
#endif

#if !_lib_wcstombs
#undef	STUB
size_t
wcstombs(char* t, register const wchar_t* s, size_t n)
{
	register char*		p = t;
	register char*		e = t + n;

	if (t)
		while (p < e && (*p++ = *s++));
	else
		while (p++, *s++);
	return p - t;
}
#endif

#if STUB
NoN(wc)
#endif
