/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 5, 2024.
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

#include "stdhdr.h"

wchar_t*
fgetws(wchar_t* s, int n, Sfio_t* f)
{
	register wchar_t*	p = s;
	register wchar_t*	e = s + n - 1;
	register wint_t		c;

	STDIO_PTR(f, "fgets", wchar_t*, (wchar_t*, int, Sfio_t*), (s, n, f))

	FWIDE(f, 0);
	while (p < e && (c = fgetwc(f)) != WEOF && (*p++ = c) != '\n');
	*p = 0;
	return s;
}

wchar_t*
getws(wchar_t* s)
{
	register wchar_t*	p = s;
	register wchar_t*	e = s + BUFSIZ - 1;
	register wint_t		c;

	FWIDE(sfstdin, 0);
	while (p < e && (c = fgetwc(sfstdin)) != WEOF && (*p++ = c) != '\n');
	*p = 0;
	return s;
}
