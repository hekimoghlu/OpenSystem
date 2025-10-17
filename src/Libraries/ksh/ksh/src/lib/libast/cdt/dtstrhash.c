/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 4, 2022.
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
#include	"dthdr.h"

/* Hashing a string into an unsigned integer.
** The basic method is to continuingly accumulate bytes and multiply
** with some given prime. The length n of the string is added last.
** The recurrent equation is like this:
**	h[k] = (h[k-1] + bytes)*prime	for 0 <= k < n
**	h[n] = (h[n-1] + n)*prime
** The prime is chosen to have a good distribution of 1-bits so that
** the multiplication will distribute the bits in the accumulator well.
** The below code accumulates 2 bytes at a time for speed.
**
** Written by Kiem-Phong Vo (02/28/03)
*/

#if __STD_C
uint dtstrhash(uint h, Void_t* args, ssize_t n)
#else
uint dtstrhash(h,args,n)
reg uint	h;
Void_t*		args;
ssize_t		n;
#endif
{
	unsigned char	*s = (unsigned char*)args;

	if(n <= 0)
	{	for(; *s != 0; s += s[1] ? 2 : 1)
			h = (h + (s[0]<<8) + s[1])*DT_PRIME;
		n = s - (unsigned char*)args;
	}
	else
	{	unsigned char*	ends;
		for(ends = s+n-1; s < ends; s += 2)
			h = (h + (s[0]<<8) + s[1])*DT_PRIME;
		if(s <= ends)
			h = (h + (s[0]<<8))*DT_PRIME;
	}
	return (h+n)*DT_PRIME;
}
