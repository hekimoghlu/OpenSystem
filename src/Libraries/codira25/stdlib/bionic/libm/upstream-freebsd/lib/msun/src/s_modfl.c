/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 15, 2023.
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
#include <float.h>
#include <math.h>
#include <sys/types.h>

#include "fpmath.h"

#if LDBL_MANL_SIZE > 32
#define	MASK	((uint64_t)-1)
#else
#define	MASK	((uint32_t)-1)
#endif
/* Return the last n bits of a word, representing the fractional part. */
#define	GETFRAC(bits, n)	((bits) & ~(MASK << (n)))
/* The number of fraction bits in manh, not counting the integer bit */
#define	HIBITS	(LDBL_MANT_DIG - LDBL_MANL_SIZE)

static const long double zero[] = { 0.0L, -0.0L };

long double
modfl(long double x, long double *iptr)
{
	union IEEEl2bits ux;
	int e;

	ux.e = x;
	e = ux.bits.exp - LDBL_MAX_EXP + 1;
	if (e < HIBITS) {			/* Integer part is in manh. */
		if (e < 0) {			/* |x|<1 */
			*iptr = zero[ux.bits.sign];
			return (x);
		} else {
			if ((GETFRAC(ux.bits.manh, HIBITS - 1 - e) |
			     ux.bits.manl) == 0) {	/* X is an integer. */
				*iptr = x;
				return (zero[ux.bits.sign]);
			} else {
				/* Clear all but the top e+1 bits. */
				ux.bits.manh >>= HIBITS - 1 - e;
				ux.bits.manh <<= HIBITS - 1 - e;
				ux.bits.manl = 0;
				*iptr = ux.e;
				return (x - ux.e);
			}
		}
	} else if (e >= LDBL_MANT_DIG - 1) {	/* x has no fraction part. */
		*iptr = x;
		if (x != x)			/* Handle NaNs. */
			return (x);
		return (zero[ux.bits.sign]);
	} else {				/* Fraction part is in manl. */
		if (GETFRAC(ux.bits.manl, LDBL_MANT_DIG - 1 - e) == 0) {
			/* x is integral. */
			*iptr = x;
			return (zero[ux.bits.sign]);
		} else {
			/* Clear all but the top e+1 bits. */
			ux.bits.manl >>= LDBL_MANT_DIG - 1 - e;
			ux.bits.manl <<= LDBL_MANT_DIG - 1 - e;
			*iptr = ux.e;
			return (x - ux.e);
		}
	}
}
