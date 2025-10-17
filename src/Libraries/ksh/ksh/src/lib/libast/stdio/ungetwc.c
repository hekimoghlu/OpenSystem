/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 2, 2025.
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

wint_t
ungetwc(wint_t c, Sfio_t* f)
{
	register unsigned char*	s = (unsigned char*)&c;
	register unsigned char*	e = s + sizeof(c);

	STDIO_INT(f, "ungetwc", wint_t, (wint_t, Sfio_t*), (c, f))

	FWIDE(f, WEOF);
	while (s < e)
		if (sfungetc(f, *s++) == EOF)
			return WEOF;
	return c;
}
