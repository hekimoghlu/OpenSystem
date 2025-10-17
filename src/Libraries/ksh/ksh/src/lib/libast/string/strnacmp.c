/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 21, 2025.
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
 * ccmapc(c, CC_NATIVE, CC_ASCII) and strncmp
 */

#include <ast.h>
#include <ccode.h>

#if _lib_strnacmp

NoN(strnacmp)

#else

#include <ctype.h>

int
strnacmp(const char* a, const char* b, size_t n)
{
#if CC_NATIVE == CC_ASCII
	return strncmp(a, b, n);
#else
	register unsigned char*	ua = (unsigned char*)a;
	register unsigned char*	ub = (unsigned char*)b;
	register unsigned char*	ue;
	register unsigned char*	m;
	register int		c;
	register int		d;

	m = ccmap(CC_NATIVE, CC_ASCII);
	ue = ua + n;
	while (ua < ue)
	{
		c = m[*ua++];
		if (d = c - m[*ub++])
			return d;
		if (!c)
			return 0;
	}
	return 0;
#endif
}

#endif
