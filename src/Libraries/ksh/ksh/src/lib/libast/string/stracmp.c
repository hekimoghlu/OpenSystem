/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 9, 2025.
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
 * ccmapchr(ccmap(CC_NATIVE,CC_ASCII),c) and strcmp
 */

#include <ast.h>
#include <ccode.h>

#if _lib_stracmp

NoN(stracmp)

#else

#include <ctype.h>

int
stracmp(const char* aa, const char* ab)
{
	register unsigned char*	a;
	register unsigned char*	b;
	register unsigned char*	m;
	register int		c;
	register int		d;

	if (!(m = ccmap(CC_NATIVE, CC_ASCII)))
		return strcmp(aa, ab);
	a = (unsigned char*)aa;
	b = (unsigned char*)ab;
	for (;;)
	{
		c = m[*a++];
		if (d = c - m[*b++])
			return d;
		if (!c)
			return 0;
	}
}

#endif
