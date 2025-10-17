/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 30, 2021.
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
#include <complex.h>
#include <float.h>

#include "fpmath.h"
#include "math.h"
#include "math_private.h"

#define	MANT_DIG	DBL_MANT_DIG
#define	MAX_EXP		DBL_MAX_EXP
#define	MIN_EXP		DBL_MIN_EXP

static const double
ln2_hi = 6.9314718055829871e-1,		/*  0x162e42fefa0000.0p-53 */
ln2_lo = 1.6465949582897082e-12;	/*  0x1cf79abc9e3b3a.0p-92 */

double complex
clog(double complex z)
{
	double_t ax, ax2h, ax2l, axh, axl, ay, ay2h, ay2l, ayh, ayl, sh, sl, t;
	double x, y, v;
	uint32_t hax, hay;
	int kx, ky;

	x = creal(z);
	y = cimag(z);
	v = atan2(y, x);

	ax = fabs(x);
	ay = fabs(y);
	if (ax < ay) {
		t = ax;
		ax = ay;
		ay = t;
	}

	GET_HIGH_WORD(hax, ax);
	kx = (hax >> 20) - 1023;
	GET_HIGH_WORD(hay, ay);
	ky = (hay >> 20) - 1023;

	/* Handle NaNs and Infs using the general formula. */
	if (kx == MAX_EXP || ky == MAX_EXP)
		return (CMPLX(log(hypot(x, y)), v));

	/* Avoid spurious underflow, and reduce inaccuracies when ax is 1. */
	if (ax == 1) {
		if (ky < (MIN_EXP - 1) / 2)
			return (CMPLX((ay / 2) * ay, v));
		return (CMPLX(log1p(ay * ay) / 2, v));
	}

	/* Avoid underflow when ax is not small.  Also handle zero args. */
	if (kx - ky > MANT_DIG || ay == 0)
		return (CMPLX(log(ax), v));

	/* Avoid overflow. */
	if (kx >= MAX_EXP - 1)
		return (CMPLX(log(hypot(x * 0x1p-1022, y * 0x1p-1022)) +
		    (MAX_EXP - 2) * ln2_lo + (MAX_EXP - 2) * ln2_hi, v));
	if (kx >= (MAX_EXP - 1) / 2)
		return (CMPLX(log(hypot(x, y)), v));

	/* Reduce inaccuracies and avoid underflow when ax is denormal. */
	if (kx <= MIN_EXP - 2)
		return (CMPLX(log(hypot(x * 0x1p1023, y * 0x1p1023)) +
		    (MIN_EXP - 2) * ln2_lo + (MIN_EXP - 2) * ln2_hi, v));

	/* Avoid remaining underflows (when ax is small but not denormal). */
	if (ky < (MIN_EXP - 1) / 2 + MANT_DIG)
		return (CMPLX(log(hypot(x, y)), v));

	/* Calculate ax*ax and ay*ay exactly using Dekker's algorithm. */
	t = (double)(ax * (0x1p27 + 1));
	axh = (double)(ax - t) + t;
	axl = ax - axh;
	ax2h = ax * ax;
	ax2l = axh * axh - ax2h + 2 * axh * axl + axl * axl;
	t = (double)(ay * (0x1p27 + 1));
	ayh = (double)(ay - t) + t;
	ayl = ay - ayh;
	ay2h = ay * ay;
	ay2l = ayh * ayh - ay2h + 2 * ayh * ayl + ayl * ayl;

	/*
	 * When log(|z|) is far from 1, accuracy in calculating the sum
	 * of the squares is not very important since log() reduces
	 * inaccuracies.  We depended on this to use the general
	 * formula when log(|z|) is very far from 1.  When log(|z|) is
	 * moderately far from 1, we go through the extra-precision
	 * calculations to reduce branches and gain a little accuracy.
	 *
	 * When |z| is near 1, we subtract 1 and use log1p() and don't
	 * leave it to log() to subtract 1, since we gain at least 1 bit
	 * of accuracy in this way.
	 *
	 * When |z| is very near 1, subtracting 1 can cancel almost
	 * 3*MANT_DIG bits.  We arrange that subtracting 1 is exact in
	 * doubled precision, and then do the rest of the calculation
	 * in sloppy doubled precision.  Although large cancellations
	 * often lose lots of accuracy, here the final result is exact
	 * in doubled precision if the large calculation occurs (because
	 * then it is exact in tripled precision and the cancellation
	 * removes enough bits to fit in doubled precision).  Thus the
	 * result is accurate in sloppy doubled precision, and the only
	 * significant loss of accuracy is when it is summed and passed
	 * to log1p().
	 */
	sh = ax2h;
	sl = ay2h;
	_2sumF(sh, sl);
	if (sh < 0.5 || sh >= 3)
		return (CMPLX(log(ay2l + ax2l + sl + sh) / 2, v));
	sh -= 1;
	_2sum(sh, sl);
	_2sum(ax2l, ay2l);
	/* Briggs-Kahan algorithm (except we discard the final low term): */
	_2sum(sh, ax2l);
	_2sum(sl, ay2l);
	t = ax2l + sl;
	_2sumF(sh, t);
	return (CMPLX(log1p(ay2l + t + sh) / 2, v));
}

#if (LDBL_MANT_DIG == 53)
__weak_reference(clog, clogl);
#endif
