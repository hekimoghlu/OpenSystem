/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 16, 2022.
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
 * copy table with element size n
 * indexed by CC_ASCII to table
 * indexed by CC_NATIVE
 */

#include <ast.h>
#include <ccode.h>

void*
ccnative(void* b, const void* a, size_t n)
{
#if CC_ASCII == CC_NATIVE
	return memcpy(b, a, n * (UCHAR_MAX + 1));
#else
	register int			c;
	register const unsigned char*	m;
	register unsigned char*		cb = (unsigned char*)b;
	register unsigned char*		ca = (unsigned char*)a;

	m = CCMAP(CC_ASCII, CC_NATIVE);
	if (n == sizeof(char))
		for (c = 0; c <= UCHAR_MAX; c++)
			cb[c] = ca[m[c]];
	else
		for (c = 0; c <= UCHAR_MAX; c++)
			memcpy(cb + n * c, ca + n * m[c], n);
	return b;
#endif
}
