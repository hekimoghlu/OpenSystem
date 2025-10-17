/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 10, 2023.
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
/* Please send bug reports to David M. Gay (dmg at acm dot org,
 * with " at " changed at "@" and " dot " changed to ".").	*/

#include "gdtoaimp.h"

 void
#ifdef KR_headers
rshift(b, k) Bigint *b; int k;
#else
rshift(Bigint *b, int k)
#endif
{
	ULong *x, *x1, *xe, y;
	int n;

	x = x1 = b->x;
	n = k >> kshift;
	if (n < b->wds) {
		xe = x + b->wds;
		x += n;
		if (k &= kmask) {
			n = ULbits - k;
			y = *x++ >> k;
			while(x < xe) {
				*x1++ = (y | (*x << n)) & ALL_ON;
				y = *x++ >> k;
				}
			if ((*x1 = y) !=0)
				x1++;
			}
		else
			while(x < xe)
				*x1++ = *x++;
		}
	if ((b->wds = x1 - b->x) == 0)
		b->x[0] = 0;
	}

 int
#ifdef KR_headers
trailz(b) Bigint *b;
#else
trailz(Bigint *b)
#endif
{
	ULong L, *x, *xe;
	int n = 0;

	x = b->x;
	xe = x + b->wds;
	for(n = 0; x < xe && !*x; x++)
		n += ULbits;
	if (x < xe) {
		L = *x;
		n += lo0bits(&L);
		}
	return n;
	}
