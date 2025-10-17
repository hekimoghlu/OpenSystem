/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 30, 2022.
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

#include "xlocale_private.h"

#include "gdtoaimp.h"

#undef _0
#undef _1

/* one or the other of IEEE_MC68k or IEEE_8087 should be #defined */

#ifdef IEEE_MC68k
#define _0 0
#define _1 1
#define _2 2
#define _3 3
#define _4 4
#endif
#ifdef IEEE_8087
#define _0 4
#define _1 3
#define _2 2
#define _3 1
#define _4 0
#endif

 int
#ifdef KR_headers
strtopx(s, sp, V, loc) CONST char *s; char **sp; void *V; locale_t loc;
#else
strtopx(CONST char *s, char **sp, void *V, locale_t loc)
#endif
{
	static FPI fpi0 = { 64, 1-16383-64+1, 32766 - 16383 - 64 + 1, 1, SI };
	ULong bits[2];
	Long exp;
	int k;
	UShort *L = (UShort*)V;
#ifdef Honor_FLT_ROUNDS
#include "gdtoa_fltrnds.h"
#else
#define fpi &fpi0
#endif

	k = strtodg(s, sp, fpi, &exp, bits, loc);
	switch(k & STRTOG_Retmask) {
	  case STRTOG_NoNumber:
		L[0] = L[1] = L[2] = L[3] = L[4] = 0;
		return k; // avoid setting sign

	  case STRTOG_Zero:
		L[0] = L[1] = L[2] = L[3] = L[4] = 0;
		break;

	  case STRTOG_Denormal:
		L[_0] = 0;
		goto normal_bits;

	  case STRTOG_Normal:
	  case STRTOG_NaNbits:
		L[_0] = exp + 0x3fff + 63;
 normal_bits:
		L[_4] = (UShort)bits[0];
		L[_3] = (UShort)(bits[0] >> 16);
		L[_2] = (UShort)bits[1];
		L[_1] = (UShort)(bits[1] >> 16);
		break;

	  case STRTOG_Infinite:
		L[_0] = 0x7fff;
		L[_1] = 0x8000;
		L[_2] = L[_3] = L[_4] = 0;
		break;

	  case STRTOG_NaN:
		L[0] = ldus_QNAN0;
		L[1] = ldus_QNAN1;
		L[2] = ldus_QNAN2;
		L[3] = ldus_QNAN3;
		L[4] = ldus_QNAN4;
	  }
	if (k & STRTOG_Neg)
		L[_0] |= 0x8000;
	return k;
	}
