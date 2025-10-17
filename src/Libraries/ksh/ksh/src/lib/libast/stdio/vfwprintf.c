/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 29, 2022.
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

int
vfwprintf(Sfio_t* f, const wchar_t* fmt, va_list args)
{
	char*	m;
	char*	x;
	wchar_t*w;
	size_t	n;
	int	v;
	Sfio_t*	t;

	STDIO_INT(f, "vfwprintf", int, (Sfio_t*, const wchar_t*, va_list), (f, fmt, args))

	FWIDE(f, WEOF);
	n = wcstombs(NiL, fmt, 0);
	if (m = malloc(n + 1))
	{
		if (t = sfstropen())
		{
			wcstombs(m, fmt, n + 1);
			sfvprintf(t, m, args);
			free(m);
			if (!(x = sfstruse(t)))
				v = -1;
			else
			{
				n = mbstowcs(NiL, x, 0);
				if (w = (wchar_t*)sfreserve(f, n * sizeof(wchar_t) + 1, 0))
					v = mbstowcs(w, x, n + 1);
				else
					v = -1;
			}
			sfstrclose(t);
		}
		else
		{
			free(m);
			v = -1;
		}
	}
	else
		v = -1;
	return v;
}
