/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 2, 2024.
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
__FBSDID("$FreeBSD: src/lib/libc/gdtoa/_ldtoa.c,v 1.5 2007/12/09 19:48:57 das Exp $");

#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include "fpmath.h"
#include "gdtoaimp.h"

/*
 * ldtoa() is a wrapper for gdtoa() that makes it smell like dtoa(),
 * except that the floating point argument is passed by reference.
 * When dtoa() is passed a NaN or infinity, it sets expt to 9999.
 * However, a long double could have a valid exponent of 9999, so we
 * use INT_MAX in ldtoa() instead.
 */
char *
__ldtoa(long double *ld, int mode, int ndigits, int *decpt, int *sign,
    char **rve)
{
#if defined(__arm__) || defined(__arm64__)
	/* On arm, double == long double, so short circuit this */
	char * ret = __dtoa((double)*ld, mode, ndigits, decpt, sign, rve);
	if (*decpt == 9999)
		*decpt = INT_MAX;
	return ret;
#else
	FPI fpi = {
		LDBL_MANT_DIG,			/* nbits */
		LDBL_MIN_EXP - LDBL_MANT_DIG,	/* emin */
		LDBL_MAX_EXP - LDBL_MANT_DIG,	/* emax */
		FLT_ROUNDS,	       		/* rounding */
#ifdef Sudden_Underflow	/* unused, but correct anyway */
		1
#else
		0
#endif
	};
	int be, kind;
	char *ret;
	union IEEEl2bits u;
	uint32_t bits[(LDBL_MANT_DIG + 31) / 32];
	void *vbits = bits;
	int type;

	u.e = *ld;
	type = fpclassify(u.e);

	/*
	 * gdtoa doesn't know anything about the sign of the number, so
	 * if the number is negative, we need to swap rounding modes of
	 * 2 (upwards) and 3 (downwards).
	 */
	*sign = u.bits.sign;
	fpi.rounding ^= (fpi.rounding >> 1) & u.bits.sign;

	be = u.bits.exp - (LDBL_MAX_EXP - 1) - (LDBL_MANT_DIG - 1);
	LDBL_TO_ARRAY32(u, bits);

	switch (type) {
	case FP_NORMAL:
	case FP_SUPERNORMAL:
		kind = STRTOG_Normal;
#ifdef	LDBL_IMPLICIT_NBIT
		bits[LDBL_MANT_DIG / 32] |= 1 << ((LDBL_MANT_DIG - 1) % 32);
#endif /* LDBL_IMPLICIT_NBIT */
		break;
	case FP_ZERO:
		kind = STRTOG_Zero;
		break;
	case FP_SUBNORMAL:
		kind = STRTOG_Denormal;
		be++;
		break;
	case FP_INFINITE:
		kind = STRTOG_Infinite;
		break;
	case FP_NAN:
		kind = STRTOG_NaN;
		break;
	default:
		LIBC_ABORT("fpclassify returned %d", type);
	}

	ret = gdtoa(&fpi, be, vbits, &kind, mode, ndigits, decpt, rve);
	if (*decpt == -32768)
		*decpt = INT_MAX;
	return ret;
#endif
}
