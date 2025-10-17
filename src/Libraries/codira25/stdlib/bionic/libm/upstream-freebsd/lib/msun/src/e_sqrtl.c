/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 27, 2024.
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
#include <fenv.h>
#include <float.h>

#include "fpmath.h"
#include "math.h"

/* Return (x + ulp) for normal positive x. Assumes no overflow. */
static inline long double
inc(long double x)
{
	union IEEEl2bits u;

	u.e = x;
	if (++u.bits.manl == 0) {
		if (++u.bits.manh == 0) {
			u.bits.exp++;
			u.bits.manh |= LDBL_NBIT;
		}
	}
	return (u.e);
}

/* Return (x - ulp) for normal positive x. Assumes no underflow. */
static inline long double
dec(long double x)
{
	union IEEEl2bits u;

	u.e = x;
	if (u.bits.manl-- == 0) {
		if (u.bits.manh-- == LDBL_NBIT) {
			u.bits.exp--;
			u.bits.manh |= LDBL_NBIT;
		}
	}
	return (u.e);
}

#pragma STDC FENV_ACCESS ON

/*
 * This is slow, but simple and portable. You should use hardware sqrt
 * if possible.
 */

long double
sqrtl(long double x)
{
	union IEEEl2bits u;
	int k, r;
	long double lo, xn;
	fenv_t env;

	u.e = x;

	/* If x = NaN, then sqrt(x) = NaN. */
	/* If x = Inf, then sqrt(x) = Inf. */
	/* If x = -Inf, then sqrt(x) = NaN. */
	if (u.bits.exp == LDBL_MAX_EXP * 2 - 1)
		return (x * x + x);

	/* If x = +-0, then sqrt(x) = +-0. */
	if ((u.bits.manh | u.bits.manl | u.bits.exp) == 0)
		return (x);

	/* If x < 0, then raise invalid and return NaN */
	if (u.bits.sign)
		return ((x - x) / (x - x));

	feholdexcept(&env);

	if (u.bits.exp == 0) {
		/* Adjust subnormal numbers. */
		u.e *= 0x1.0p514;
		k = -514;
	} else {
		k = 0;
	}
	/*
	 * u.e is a normal number, so break it into u.e = e*2^n where
	 * u.e = (2*e)*2^2k for odd n and u.e = (4*e)*2^2k for even n.
	 */
	if ((u.bits.exp - 0x3ffe) & 1) {	/* n is odd.     */
		k += u.bits.exp - 0x3fff;	/* 2k = n - 1.   */
		u.bits.exp = 0x3fff;		/* u.e in [1,2). */
	} else {
		k += u.bits.exp - 0x4000;	/* 2k = n - 2.   */
		u.bits.exp = 0x4000;		/* u.e in [2,4). */
	}

	/*
	 * Newton's iteration.
	 * Split u.e into a high and low part to achieve additional precision.
	 */
	xn = sqrt(u.e);			/* 53-bit estimate of sqrtl(x). */
#if LDBL_MANT_DIG > 100
	xn = (xn + (u.e / xn)) * 0.5;	/* 106-bit estimate. */
#endif
	lo = u.e;
	u.bits.manl = 0;		/* Zero out lower bits. */
	lo = (lo - u.e) / xn;		/* Low bits divided by xn. */
	xn = xn + (u.e / xn);		/* High portion of estimate. */
	u.e = xn + lo;			/* Combine everything. */
	u.bits.exp += (k >> 1) - 1;

	feclearexcept(FE_INEXACT);
	r = fegetround();
	fesetround(FE_TOWARDZERO);	/* Set to round-toward-zero. */
	xn = x / u.e;			/* Chopped quotient (inexact?). */

	if (!fetestexcept(FE_INEXACT)) { /* Quotient is exact. */
		if (xn == u.e) {
			fesetenv(&env);
			return (u.e);
		}
		/* Round correctly for inputs like x = y**2 - ulp. */
		xn = dec(xn);		/* xn = xn - ulp. */
	}

	if (r == FE_TONEAREST) {
		xn = inc(xn);		/* xn = xn + ulp. */
	} else if (r == FE_UPWARD) {
		u.e = inc(u.e);		/* u.e = u.e + ulp. */
		xn = inc(xn);		/* xn  = xn + ulp. */
	}
	u.e = u.e + xn;				/* Chopped sum. */
	feupdateenv(&env);	/* Restore env and raise inexact */
	u.bits.exp--;
	return (u.e);
}
