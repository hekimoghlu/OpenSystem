/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 21, 2022.
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
/**
 * sinpi(x) computes sin(pi*x) without multiplication by pi (almost).  First,
 * note that sinpi(-x) = -sinpi(x), so the algorithm considers only |x| and
 * includes reflection symmetry by considering the sign of x on output.  The
 * method used depends on the magnitude of x.
 *
 * 1. For small |x|, sinpi(x) = pi * x where a sloppy threshold is used.  The
 *    threshold is |x| < 0x1pN with N = -(P/2+M).  P is the precision of the
 *    floating-point type and M = 2 to 4.  To achieve high accuracy, pi is 
 *    decomposed into high and low parts with the high part containing a
 *    number of trailing zero bits.  x is also split into high and low parts.
 *
 * 2. For |x| < 1, argument reduction is not required and sinpi(x) is 
 *    computed by calling a kernel that leverages the kernels for sin(x)
 *    ans cos(x).  See k_sinpi.c and k_cospi.c for details.
 *
 * 3. For 1 <= |x| < 0x1p(P-1), argument reduction is required where
 *    |x| = j0 + r with j0 an integer and the remainder r satisfies
 *    0 <= r < 1.  With the given domain, a simplified inline floor(x)
 *    is used.  Also, note the following identity
 *
 *    sinpi(x) = sin(pi*(j0+r))
 *             = sin(pi*j0) * cos(pi*r) + cos(pi*j0) * sin(pi*r)
 *             = cos(pi*j0) * sin(pi*r)
 *             = +-sinpi(r)
 *
 *    If j0 is even, then cos(pi*j0) = 1. If j0 is odd, then cos(pi*j0) = -1.
 *    sinpi(r) is then computed via an appropriate kernel.
 *
 * 4. For |x| >= 0x1p(P-1), |x| is integral and sinpi(x) = copysign(0,x).
 *
 * 5. Special cases:
 *
 *    sinpi(+-0) = +-0
 *    sinpi(+-n) = +-0, for positive integers n.
 *    sinpi(+-inf) = nan.  Raises the "invalid" floating-point exception.
 *    sinpi(nan) = nan.  Raises the "invalid" floating-point exception.
 */

#include <float.h>
#include "math.h"
#include "math_private.h"

static const double
pi_hi = 3.1415926814079285e+00,	/* 0x400921fb 0x58000000 */
pi_lo =-2.7818135228334233e-08;	/* 0xbe5dde97 0x3dcb3b3a */

#include "k_cospi.h"
#include "k_sinpi.h"

volatile static const double vzero = 0;

double
sinpi(double x)
{
	double ax, hi, lo, s;
	uint32_t hx, ix, j0, lx;

	EXTRACT_WORDS(hx, lx, x);
	ix = hx & 0x7fffffff;
	INSERT_WORDS(ax, ix, lx);

	if (ix < 0x3ff00000) {			/* |x| < 1 */
		if (ix < 0x3fd00000) {		/* |x| < 0.25 */
			if (ix < 0x3e200000) {	/* |x| < 0x1p-29 */
				if (x == 0)
					return (x);
				/*
				 * To avoid issues with subnormal values,
				 * scale the computation and rescale on 
				 * return.
				 */
				INSERT_WORDS(hi, hx, 0);
				hi *= 0x1p53;
				lo = x * 0x1p53 - hi;
				s = (pi_lo + pi_hi) * lo + pi_lo * hi +
				    pi_hi * hi;
				return (s * 0x1p-53);
			}

			s = __kernel_sinpi(ax);
			return ((hx & 0x80000000) ? -s : s);
		}

		if (ix < 0x3fe00000)		/* |x| < 0.5 */
			s = __kernel_cospi(0.5 - ax);
		else if (ix < 0x3fe80000)	/* |x| < 0.75 */
			s = __kernel_cospi(ax - 0.5);
		else
			s = __kernel_sinpi(1 - ax);
		return ((hx & 0x80000000) ? -s : s);
	}

	if (ix < 0x43300000) {			/* 1 <= |x| < 0x1p52 */
		FFLOOR(x, j0, ix, lx);	/* Integer part of ax. */
		ax -= x;
		EXTRACT_WORDS(ix, lx, ax);

		if (ix == 0)
			s = 0;
		else {
			if (ix < 0x3fe00000) {		/* |x| < 0.5 */
				if (ix < 0x3fd00000)	/* |x| < 0.25 */
					s = __kernel_sinpi(ax);
				else 
					s = __kernel_cospi(0.5 - ax);
			} else {
				if (ix < 0x3fe80000)	/* |x| < 0.75 */
					s = __kernel_cospi(ax - 0.5);
				else
					s = __kernel_sinpi(1 - ax);
			}

			if (j0 > 30)
				x -= 0x1p30;
			j0 = (uint32_t)x;
			if (j0 & 1) s = -s;
		}

		return ((hx & 0x80000000) ? -s : s);
	}

	/* x = +-inf or nan. */
	if (ix >= 0x7ff00000)
		return (vzero / vzero);

	/*
	 * |x| >= 0x1p52 is always an integer, so return +-0.
	 */
	return (copysign(0, x));
}

#if LDBL_MANT_DIG == 53
__weak_reference(sinpi, sinpil);
#endif
