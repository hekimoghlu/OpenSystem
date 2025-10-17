/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 16, 2022.
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
__FBSDID("$FreeBSD: src/lib/libc/gdtoa/_hdtoa.c,v 1.5 2007/05/08 02:59:37 das Exp $");

#include <float.h>
#include <limits.h>
#include <math.h>
#include "fpmath.h"
#include "gdtoaimp.h"

/* Strings values used by dtoa() */
#define	INFSTR	"Infinity"
#define	NANSTR	"NaN"

#define	DBL_ADJ		(DBL_MAX_EXP - 2 + ((DBL_MANT_DIG - 1) % 4))
#define	LDBL_ADJ	(LDBL_MAX_EXP - 2 + ((LDBL_MANT_DIG - 1) % 4))

/*
 * Round up the given digit string.  If the digit string is fff...f,
 * this procedure sets it to 100...0 and returns 1 to indicate that
 * the exponent needs to be bumped.  Otherwise, 0 is returned.
 */
static int
roundup(char *s0, int ndigits)
{
	char *s;

	for (s = s0 + ndigits - 1; *s == 0xf; s--) {
		if (s == s0) {
			*s = 1;
			return (1);
		}
		*s = 0;
	}
	++*s;
	return (0);
}

/*
 * Round the given digit string to ndigits digits according to the
 * current rounding mode.  Note that this could produce a string whose
 * value is not representable in the corresponding floating-point
 * type.  The exponent pointed to by decpt is adjusted if necessary.
 */
static void
dorounding(char *s0, int ndigits, int sign, int *decpt)
{
	int adjust = 0;	/* do we need to adjust the exponent? */

	switch (FLT_ROUNDS) {
	case 0:		/* toward zero */
	default:	/* implementation-defined */
		break;
	case 1:		/* to nearest, halfway rounds to even */
		if ((s0[ndigits] > 8) ||
		    (s0[ndigits] == 8 && s0[ndigits + 1] & 1))
			adjust = roundup(s0, ndigits);
		break;
	case 2:		/* toward +inf */
		if (sign == 0)
			adjust = roundup(s0, ndigits);
		break;
	case 3:		/* toward -inf */
		if (sign != 0)
			adjust = roundup(s0, ndigits);
		break;
	}

	if (adjust)
		*decpt += 4;
}

/*
 * This procedure converts a double-precision number in IEEE format
 * into a string of hexadecimal digits and an exponent of 2.  Its
 * behavior is bug-for-bug compatible with dtoa() in mode 2, with the
 * following exceptions:
 *
 * - An ndigits < 0 causes it to use as many digits as necessary to
 *   represent the number exactly.
 * - The additional xdigs argument should point to either the string
 *   "0123456789ABCDEF" or the string "0123456789abcdef", depending on
 *   which case is desired.
 * - This routine does not repeat dtoa's mistake of setting decpt
 *   to 9999 in the case of an infinity or NaN.  INT_MAX is used
 *   for this purpose instead.
 *
 * Note that the C99 standard does not specify what the leading digit
 * should be for non-zero numbers.  For instance, 0x1.3p3 is the same
 * as 0x2.6p2 is the same as 0x4.cp3.  This implementation chooses the
 * first digit so that subsequent digits are aligned on nibble
 * boundaries (before rounding).
 *
 * Inputs:	d, xdigs, ndigits
 * Outputs:	decpt, sign, rve
 */
char *
__hdtoa(double d, const char *xdigs, int ndigits, int *decpt, int *sign,
    char **rve)
{
	static const int sigfigs = (DBL_MANT_DIG + 3) / 4;
	union IEEEd2bits u;
	char *s, *s0;
	int bufsize, f;

	u.d = d;
	*sign = u.bits.sign;

	switch (f = fpclassify(d)) {
	case FP_NORMAL:
		*decpt = u.bits.exp - DBL_ADJ;
		break;
	case FP_ZERO:
return_zero:
		*decpt = 1;
		return (nrv_alloc("0", rve, 1));
	case FP_SUBNORMAL:
		/*
		 * For processors that treat subnormals as zero, comparison
		 * with zero will be equal, so we jump to the FP_ZERO case.
		 */
		if(u.d == 0.0) goto return_zero;
		u.d *= 0x1p514;
		*decpt = u.bits.exp - (514 + DBL_ADJ);
		break;
	case FP_INFINITE:
		*decpt = INT_MAX;
		return (nrv_alloc(INFSTR, rve, sizeof(INFSTR) - 1));
	case FP_NAN:
		*decpt = INT_MAX;
		return (nrv_alloc(NANSTR, rve, sizeof(NANSTR) - 1));
	default:
		LIBC_ABORT("fpclassify returned %d", f);
	}

	/* FP_NORMAL or FP_SUBNORMAL */

	if (ndigits == 0)		/* dtoa() compatibility */
		ndigits = 1;

	/*
	 * For simplicity, we generate all the digits even if the
	 * caller has requested fewer.
	 */
	bufsize = (sigfigs > ndigits) ? sigfigs : ndigits;
	s0 = rv_alloc(bufsize);

	/*
	 * We work from right to left, first adding any requested zero
	 * padding, then the least significant portion of the
	 * mantissa, followed by the most significant.  The buffer is
	 * filled with the byte values 0x0 through 0xf, which are
	 * converted to xdigs[0x0] through xdigs[0xf] after the
	 * rounding phase.
	 */
	for (s = s0 + bufsize - 1; s > s0 + sigfigs - 1; s--)
		*s = 0;
	for (; s > s0 + sigfigs - (DBL_MANL_SIZE / 4) - 1 && s > s0; s--) {
		*s = u.bits.manl & 0xf;
		u.bits.manl >>= 4;
	}
	for (; s > s0; s--) {
		*s = u.bits.manh & 0xf;
		u.bits.manh >>= 4;
	}

	/*
	 * At this point, we have snarfed all the bits in the
	 * mantissa, with the possible exception of the highest-order
	 * (partial) nibble, which is dealt with by the next
	 * statement.  We also tack on the implicit normalization bit.
	 */
	*s = u.bits.manh | (1U << ((DBL_MANT_DIG - 1) % 4));

	/* If ndigits < 0, we are expected to auto-size the precision. */
	if (ndigits < 0) {
		for (ndigits = sigfigs; s0[ndigits - 1] == 0; ndigits--)
			;
	}

	if (sigfigs > ndigits && s0[ndigits] != 0)
		dorounding(s0, ndigits, u.bits.sign, decpt);

	s = s0 + ndigits;
	if (rve != NULL)
		*rve = s;
	*s-- = '\0';
	for (; s >= s0; s--)
		*s = xdigs[(unsigned int)*s];

	return (s0);
}

#if (LDBL_MANT_DIG > DBL_MANT_DIG)

/*
 * This is the long double version of __hdtoa().
 */
char *
__hldtoa(long double e, const char *xdigs, int ndigits, int *decpt, int *sign,
    char **rve)
{
	static const int sigfigs = (LDBL_MANT_DIG + 3) / 4;
	union IEEEl2bits u;
	char *s, *s0;
	int bufsize, f;

	u.e = e;
	*sign = u.bits.sign;

	switch (f = fpclassify(e)) {
	case FP_NORMAL:
	case FP_SUPERNORMAL:
		*decpt = u.bits.exp - LDBL_ADJ;
		break;
	case FP_ZERO:
		*decpt = 1;
		return (nrv_alloc("0", rve, 1));
	case FP_SUBNORMAL:
		u.e *= 0x1p514L;
		*decpt = u.bits.exp - (514 + LDBL_ADJ);
		break;
	case FP_INFINITE:
		*decpt = INT_MAX;
		return (nrv_alloc(INFSTR, rve, sizeof(INFSTR) - 1));
	case FP_NAN:
		*decpt = INT_MAX;
		return (nrv_alloc(NANSTR, rve, sizeof(NANSTR) - 1));
	default:
		LIBC_ABORT("fpclassify returned %d", f);
	}

	/* FP_NORMAL or FP_SUBNORMAL */

	if (ndigits == 0)		/* dtoa() compatibility */
		ndigits = 1;

	/*
	 * For simplicity, we generate all the digits even if the
	 * caller has requested fewer.
	 */
	bufsize = (sigfigs > ndigits) ? sigfigs : ndigits;
	s0 = rv_alloc(bufsize);

	/*
	 * We work from right to left, first adding any requested zero
	 * padding, then the least significant portion of the
	 * mantissa, followed by the most significant.  The buffer is
	 * filled with the byte values 0x0 through 0xf, which are
	 * converted to xdigs[0x0] through xdigs[0xf] after the
	 * rounding phase.
	 */
	for (s = s0 + bufsize - 1; s > s0 + sigfigs - 1; s--)
		*s = 0;
	for (; s > s0 + sigfigs - (LDBL_MANL_SIZE / 4) - 1 && s > s0; s--) {
		*s = u.bits.manl & 0xf;
		u.bits.manl >>= 4;
	}
	for (; s > s0; s--) {
		*s = u.bits.manh & 0xf;
		u.bits.manh >>= 4;
	}

	/*
	 * At this point, we have snarfed all the bits in the
	 * mantissa, with the possible exception of the highest-order
	 * (partial) nibble, which is dealt with by the next
	 * statement.  We also tack on the implicit normalization bit.
	 */
	*s = u.bits.manh | (1U << ((LDBL_MANT_DIG - 1) % 4));

	/* If ndigits < 0, we are expected to auto-size the precision. */
	if (ndigits < 0) {
		for (ndigits = sigfigs; s0[ndigits - 1] == 0; ndigits--)
			;
	}

	if (sigfigs > ndigits && s0[ndigits] != 0)
		dorounding(s0, ndigits, u.bits.sign, decpt);

	s = s0 + ndigits;
	if (rve != NULL)
		*rve = s;
	*s-- = '\0';
	for (; s >= s0; s--)
		*s = xdigs[(unsigned int)*s];

	return (s0);
}

#else	/* (LDBL_MANT_DIG == DBL_MANT_DIG) */

char *
__hldtoa(long double e, const char *xdigs, int ndigits, int *decpt, int *sign,
    char **rve)
{

	return (__hdtoa((double)e, xdigs, ndigits, decpt, sign, rve));
}

#endif	/* (LDBL_MANT_DIG == DBL_MANT_DIG) */
