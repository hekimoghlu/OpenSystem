/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 26, 2022.
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
 * Hyperbolic tangent of a complex argument z = x + I y.
 *
 * The algorithm is from:
 *
 *   W. Kahan.  Branch Cuts for Complex Elementary Functions or Much
 *   Ado About Nothing's Sign Bit.  In The State of the Art in
 *   Numerical Analysis, pp. 165 ff.  Iserles and Powell, eds., 1987.
 *
 * Method:
 *
 *   Let t    = tan(x)
 *       beta = 1/cos^2(y)
 *       s    = sinh(x)
 *       rho  = cosh(x)
 *
 *   We have:
 *
 *   tanh(z) = sinh(z) / cosh(z)
 *
 *             sinh(x) cos(y) + I cosh(x) sin(y)
 *           = ---------------------------------
 *             cosh(x) cos(y) + I sinh(x) sin(y)
 *
 *             cosh(x) sinh(x) / cos^2(y) + I tan(y)
 *           = -------------------------------------
 *                    1 + sinh^2(x) / cos^2(y)
 *
 *             beta rho s + I t
 *           = ----------------
 *               1 + beta s^2
 *
 * Modifications:
 *
 *   I omitted the original algorithm's handling of overflow in tan(x) after
 *   verifying with nearpi.c that this can't happen in IEEE single or double
 *   precision.  I also handle large x differently.
 */

#include <complex.h>
#include <math.h>

#include "math_private.h"

double complex
ctanh(double complex z)
{
	double x, y;
	double t, beta, s, rho, denom;
	uint32_t hx, ix, lx;

	x = creal(z);
	y = cimag(z);

	EXTRACT_WORDS(hx, lx, x);
	ix = hx & 0x7fffffff;

	/*
	 * ctanh(NaN +- I 0) = d(NaN) +- I 0
	 *
	 * ctanh(NaN + I y) = d(NaN,y) + I d(NaN,y)	for y != 0
	 *
	 * The imaginary part has the sign of x*sin(2*y), but there's no
	 * special effort to get this right.
	 *
	 * ctanh(+-Inf +- I Inf) = +-1 +- I 0
	 *
	 * ctanh(+-Inf + I y) = +-1 + I 0 sin(2y)	for y finite
	 *
	 * The imaginary part of the sign is unspecified.  This special
	 * case is only needed to avoid a spurious invalid exception when
	 * y is infinite.
	 */
	if (ix >= 0x7ff00000) {
		if ((ix & 0xfffff) | lx)	/* x is NaN */
			return (CMPLX(nan_mix(x, y),
			    y == 0 ? y : nan_mix(x, y)));
		SET_HIGH_WORD(x, hx - 0x40000000);	/* x = copysign(1, x) */
		return (CMPLX(x, copysign(0, isinf(y) ? y : sin(y) * cos(y))));
	}

	/*
	 * ctanh(+-0 + i NAN) = +-0 + i NaN
	 * ctanh(+-0 +- i Inf) = +-0 + i NaN
	 * ctanh(x + i NAN) = NaN + i NaN
	 * ctanh(x +- i Inf) = NaN + i NaN
	 */
	if (!isfinite(y))
		return (CMPLX(x ? y - y : x, y - y));

	/*
	 * ctanh(+-huge +- I y) ~= +-1 +- I 2sin(2y)/exp(2x), using the
	 * approximation sinh^2(huge) ~= exp(2*huge) / 4.
	 * We use a modified formula to avoid spurious overflow.
	 */
	if (ix >= 0x40360000) {	/* |x| >= 22 */
		double exp_mx = exp(-fabs(x));
		return (CMPLX(copysign(1, x),
		    4 * sin(y) * cos(y) * exp_mx * exp_mx));
	}

	/* Kahan's algorithm */
	t = tan(y);
	beta = 1.0 + t * t;	/* = 1 / cos^2(y) */
	s = sinh(x);
	rho = sqrt(1 + s * s);	/* = cosh(x) */
	denom = 1 + beta * s * s;
	return (CMPLX((beta * rho * s) / denom, t / denom));
}

double complex
ctan(double complex z)
{

	/* ctan(z) = -I * ctanh(I * z) = I * conj(ctanh(I * conj(z))) */
	z = ctanh(CMPLX(cimag(z), creal(z)));
	return (CMPLX(cimag(z), creal(z)));
}
