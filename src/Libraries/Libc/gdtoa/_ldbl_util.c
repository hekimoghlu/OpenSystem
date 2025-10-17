/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 25, 2023.
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
#include <sys/cdefs.h>
#include <stdint.h>
#include <strings.h>
#include <float.h>
#include <math.h>

#include "fpmath.h"

#define	BITS64		64
#define	DBL_BIAS	(DBL_MAX_EXP - 1)
#define	DBL_SUBEXP	(-DBL_BIAS + 1)
#define	LL_BITS		(8 * sizeof(int64_t))
#define	LL_HIGHBIT	(1LL << 63)

__private_extern__ int
_ldbl2array32dd(union IEEEl2bits u, uint32_t *a)
{
	int bit, shift, highbit, dexp;
	uint64_t a64[2];
	int64_t t64;
	int extrabit = 0;

	if(u.d[0] == 0.0) {
	    a[0] = a[1] = a[2] = a[3] = 0;
	    return 0;
	}

	bzero(a64, sizeof(a64));

	switch (__fpclassifyd(u.d[0])) {
	case FP_NORMAL:
		/*
		 * special case: if the head double only has the high (hidden)
		 * bit set, and the tail double is non-zero and is opposite
		 * in sign, then we increment extrabit to keep 106 bit
		 * precision in the results.
		 */
		if(u.bits.manh == 0 && u.d[1] != 0 && u.bits.sign != u.bits.sign2)
		    extrabit++;
		a64[1] = (1LL << (LDBL_MANT_DIG - BITS64 - 1 + extrabit));
		a64[1] |= ((uint64_t)u.bits.manh >> (BITS64 - LDBL_MANL_SIZE - extrabit));
		a64[0] = ((uint64_t)u.bits.manh << (LDBL_MANL_SIZE + extrabit));
		break;
	case FP_SUBNORMAL:
		a64[1] |= ((uint64_t)u.bits.manh >> (BITS64 - LDBL_MANL_SIZE));
		a64[0] = ((uint64_t)u.bits.manh << LDBL_MANL_SIZE);
		/* the tail double will be zero, so we are done */
		goto done;
	default:
		goto done;
	}

	dexp = (int)u.bits.exp - (int)u.bits.exp2;
	/*
	 * if the tail double is so small to not fit in LDBL_MANT_DIG bits,
	 * then just skip it.
	 */
	if (dexp >= LDBL_MANT_DIG + extrabit) {
		reshift:
		if (extrabit) {
			bit = a64[1] & 1;
			a64[1] >>= 1;
			a64[0] >>= 1;
			a64[0] |= ((uint64_t)bit) << (BITS64 - 1);
			extrabit = 0;
		}
		goto done;
	}

	switch (__fpclassifyd(u.d[1])) {
	case FP_NORMAL:
		bit = LDBL_MANT_DIG - dexp - 1 + extrabit;
		t64 = (1LL << bit);
		break;
	case FP_SUBNORMAL:
		bit = LDBL_MANT_DIG - (int)u.bits.exp + extrabit;
		t64 = 0;
		break;
	default:
		/* should never get here */
		goto reshift;
	}
	shift = LDBL_MANL_SIZE - bit - 1;
	if (shift >= 0)
		t64 |= (u.bits.manl >> shift);
	else
		t64 |= (u.bits.manl << (-shift));
	highbit = ((a64[0] & LL_HIGHBIT) != 0);
	if (u.bits.sign == u.bits.sign2) {
		a64[0] += t64;
		if (highbit && !(a64[0] & LL_HIGHBIT))	/* carry */
			a64[1]++;
	} else {
		a64[0] -= t64;
		if (!highbit && (a64[0] & LL_HIGHBIT))	/* borrow */
			a64[1]--;
	}

	done:
	a[0] = (uint32_t)a64[0];
	a[1] = (uint32_t)(a64[0] >> 32);
	a[2] = (uint32_t)a64[1];
	a[3] = (uint32_t)(a64[1] >> 32);
	return extrabit;
}
