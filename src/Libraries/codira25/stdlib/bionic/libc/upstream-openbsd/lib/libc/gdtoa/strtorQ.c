/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 5, 2025.
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

#undef _0
#undef _1

/* one or the other of IEEE_MC68k or IEEE_8087 should be #defined */

#ifdef IEEE_MC68k
#define _0 0
#define _1 1
#define _2 2
#define _3 3
#endif
#ifdef IEEE_8087
#define _0 3
#define _1 2
#define _2 1
#define _3 0
#endif

 void
#ifdef KR_headers
ULtoQ(L, bits, exp, k) ULong *L; ULong *bits; Long exp; int k;
#else
ULtoQ(ULong *L, ULong *bits, Long exp, int k)
#endif
{
	switch(k & STRTOG_Retmask) {
	  case STRTOG_NoNumber:
	  case STRTOG_Zero:
		L[0] = L[1] = L[2] = L[3] = 0;
		break;

	  case STRTOG_Normal:
	  case STRTOG_NaNbits:
		L[_3] = bits[0];
		L[_2] = bits[1];
		L[_1] = bits[2];
		L[_0] = (bits[3] & ~0x10000) | ((exp + 0x3fff + 112) << 16);
		break;

	  case STRTOG_Denormal:
		L[_3] = bits[0];
		L[_2] = bits[1];
		L[_1] = bits[2];
		L[_0] = bits[3];
		break;

	  case STRTOG_NoMemory:
		errno = ERANGE;
		/* FALLTHROUGH */
	  case STRTOG_Infinite:
		L[_0] = 0x7fff0000;
		L[_1] = L[_2] = L[_3] = 0;
		break;

	  case STRTOG_NaN:
		L[0] = ld_QNAN0;
		L[1] = ld_QNAN1;
		L[2] = ld_QNAN2;
		L[3] = ld_QNAN3;
	  }
	if (k & STRTOG_Neg)
		L[_0] |= 0x80000000L;
	}

 int
#ifdef KR_headers
strtorQ(s, sp, rounding, L) CONST char *s; char **sp; int rounding; void *L;
#else
strtorQ(CONST char *s, char **sp, int rounding, void *L)
#endif
{
	static FPI fpi0 = { 113, 1-16383-113+1, 32766-16383-113+1, 1, SI };
	FPI *fpi, fpi1;
	ULong bits[4];
	Long exp;
	int k;

	fpi = &fpi0;
	if (rounding != FPI_Round_near) {
		fpi1 = fpi0;
		fpi1.rounding = rounding;
		fpi = &fpi1;
		}
	k = strtodg(s, sp, fpi, &exp, bits);
	ULtoQ((ULong*)L, bits, exp, k);
	return k;
	}
