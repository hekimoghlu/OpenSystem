/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 19, 2024.
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
/*
 * floorl(x)
 * Return x rounded toward -inf to integral value
 * Method:
 *	Bit twiddling.
 * Exception:
 *	Inexact flag raised if x not equal to floorl(x).
 */

#include <float.h>
#include <math.h>
#include <stdint.h>

#include "fpmath.h"

#ifdef LDBL_IMPLICIT_NBIT
#define	MANH_SIZE	(LDBL_MANH_SIZE + 1)
#define	INC_MANH(u, c)	do {					\
	uint64_t o = u.bits.manh;				\
	u.bits.manh += (c);					\
	if (u.bits.manh < o)					\
		u.bits.exp++;					\
} while (0)
#else
#define	MANH_SIZE	LDBL_MANH_SIZE
#define	INC_MANH(u, c)	do {					\
	uint64_t o = u.bits.manh;				\
	u.bits.manh += (c);					\
	if (u.bits.manh < o) {					\
		u.bits.exp++;					\
		u.bits.manh |= 1llu << (LDBL_MANH_SIZE - 1);	\
	}							\
} while (0)
#endif

static const long double huge = 1.0e300;

long double
floorl(long double x)
{
	union IEEEl2bits u = { .e = x };
	int e = u.bits.exp - LDBL_MAX_EXP + 1;

	if (e < MANH_SIZE - 1) {
		if (e < 0) {			/* raise inexact if x != 0 */
			if (huge + x > 0.0)
				if (u.bits.exp > 0 ||
				    (u.bits.manh | u.bits.manl) != 0)
					u.e = u.bits.sign ? -1.0 : 0.0;
		} else {
			uint64_t m = ((1llu << MANH_SIZE) - 1) >> (e + 1);
			if (((u.bits.manh & m) | u.bits.manl) == 0)
				return (x);	/* x is integral */
			if (u.bits.sign) {
#ifdef LDBL_IMPLICIT_NBIT
				if (e == 0)
					u.bits.exp++;
				else
#endif
				INC_MANH(u, 1llu << (MANH_SIZE - e - 1));
			}
			if (huge + x > 0.0) {	/* raise inexact flag */
				u.bits.manh &= ~m;
				u.bits.manl = 0;
			}
		}
	} else if (e < LDBL_MANT_DIG - 1) {
		uint64_t m = (uint64_t)-1 >> (64 - LDBL_MANT_DIG + e + 1);
		if ((u.bits.manl & m) == 0)
			return (x);	/* x is integral */
		if (u.bits.sign) {
			if (e == MANH_SIZE - 1)
				INC_MANH(u, 1);
			else {
				uint64_t o = u.bits.manl;
				u.bits.manl += 1llu << (LDBL_MANT_DIG - e - 1);
				if (u.bits.manl < o)	/* got a carry */
					INC_MANH(u, 1);
			}
		}
		if (huge + x > 0.0)		/* raise inexact flag */
			u.bits.manl &= ~m;
	}
	return (u.e);
}
