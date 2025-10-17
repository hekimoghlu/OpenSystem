/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 14, 2025.
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

 float
#ifdef KR_headers
strtof_l(s, sp, loc) CONST char *s; char **sp; locale_t loc;
#else
strtof_l(CONST char *s, char **sp, locale_t loc)
#endif
{
	static CONST FPI fpi0 = { 24, 1-127-24+1,  254-127-24+1, 1, SI };
	ULong bits[1];
	Long exp;
	int k;
	union { ULong L[1]; float f; } u;
#ifdef Honor_FLT_ROUNDS
#include "gdtoa_fltrnds.h"
#else
#define fpi &fpi0
#endif

	NORMALIZE_LOCALE(loc);
	k = strtodg(s, sp, fpi, &exp, bits, loc);
	switch(k & STRTOG_Retmask) {
	  case STRTOG_NoNumber:
		u.L[0] = 0;
		return u.f; // avoid setting sign

	  case STRTOG_Zero:
		u.L[0] = 0;
		break;

	  case STRTOG_Normal:
	  case STRTOG_NaNbits:
		u.L[0] = (bits[0] & 0x7fffff) | ((exp + 0x7f + 23) << 23);
		break;

	  case STRTOG_Denormal:
		u.L[0] = bits[0];
		break;

	  case STRTOG_Infinite:
		u.L[0] = 0x7f800000;
		break;

	  case STRTOG_NaN:
		u.L[0] = f_QNAN;
	  }
	if (k & STRTOG_Neg)
		u.L[0] |= 0x80000000L;
	return u.f;
	}

 float
#ifdef KR_headers
strtof(s, sp) CONST char *s; char **sp;
#else
strtof(CONST char *s, char **sp)
#endif
{
	return strtof_l(s, sp, __current_locale());
}
