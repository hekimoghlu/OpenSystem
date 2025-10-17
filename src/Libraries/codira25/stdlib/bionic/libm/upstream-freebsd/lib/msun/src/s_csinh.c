/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 25, 2024.
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
 * Hyperbolic sine of a complex argument z = x + i y.
 *
 * sinh(z) = sinh(x+iy)
 *         = sinh(x) cos(y) + i cosh(x) sin(y).
 *
 * Exceptional values are noted in the comments within the source code.
 * These values and the return value were taken from n1124.pdf.
 * The sign of the result for some exceptional values is unspecified but
 * must satisfy both sinh(conj(z)) == conj(sinh(z)) and sinh(-z) == -sinh(z).
 */

#include <complex.h>
#include <math.h>

#include "math_private.h"

static const double huge = 0x1p1023;

double complex
csinh(double complex z)
{
	double x, y, h;
	int32_t hx, hy, ix, iy, lx, ly;

	x = creal(z);
	y = cimag(z);

	EXTRACT_WORDS(hx, lx, x);
	EXTRACT_WORDS(hy, ly, y);

	ix = 0x7fffffff & hx;
	iy = 0x7fffffff & hy;

	/* Handle the nearly-non-exceptional cases where x and y are finite. */
	if (ix < 0x7ff00000 && iy < 0x7ff00000) {
		if ((iy | ly) == 0)
			return (CMPLX(sinh(x), y));
		if (ix < 0x40360000)	/* |x| < 22: normal case */
			return (CMPLX(sinh(x) * cos(y), cosh(x) * sin(y)));

		/* |x| >= 22, so cosh(x) ~= exp(|x|) */
		if (ix < 0x40862e42) {
			/* x < 710: exp(|x|) won't overflow */
			h = exp(fabs(x)) * 0.5;
			return (CMPLX(copysign(h, x) * cos(y), h * sin(y)));
		} else if (ix < 0x4096bbaa) {
			/* x < 1455: scale to avoid overflow */
			z = __ldexp_cexp(CMPLX(fabs(x), y), -1);
			return (CMPLX(creal(z) * copysign(1, x), cimag(z)));
		} else {
			/* x >= 1455: the result always overflows */
			h = huge * x;
			return (CMPLX(h * cos(y), h * h * sin(y)));
		}
	}

	/*
	 * sinh(+-0 +- I Inf) = +-0 + I dNaN.
	 * The sign of 0 in the result is unspecified.  Choice = same sign
	 * as the argument.  Raise the invalid floating-point exception.
	 *
	 * sinh(+-0 +- I NaN) = +-0 + I d(NaN).
	 * The sign of 0 in the result is unspecified.  Choice = same sign
	 * as the argument.
	 */
	if ((ix | lx) == 0)		/* && iy >= 0x7ff00000 */
		return (CMPLX(x, y - y));

	/*
	 * sinh(+-Inf +- I 0) = +-Inf + I +-0.
	 *
	 * sinh(NaN +- I 0)   = d(NaN) + I +-0.
	 */
	if ((iy | ly) == 0)		/* && ix >= 0x7ff00000 */
		return (CMPLX(x + x, y));

	/*
	 * sinh(x +- I Inf) = dNaN + I dNaN.
	 * Raise the invalid floating-point exception for finite nonzero x.
	 *
	 * sinh(x + I NaN) = d(NaN) + I d(NaN).
	 * Optionally raises the invalid floating-point exception for finite
	 * nonzero x.  Choice = don't raise (except for signaling NaNs).
	 */
	if (ix < 0x7ff00000)		/* && iy >= 0x7ff00000 */
		return (CMPLX(y - y, y - y));

	/*
	 * sinh(+-Inf + I NaN)  = +-Inf + I d(NaN).
	 * The sign of Inf in the result is unspecified.  Choice = same sign
	 * as the argument.
	 *
	 * sinh(+-Inf +- I Inf) = +-Inf + I dNaN.
	 * The sign of Inf in the result is unspecified.  Choice = same sign
	 * as the argument.  Raise the invalid floating-point exception.
	 *
	 * sinh(+-Inf + I y)   = +-Inf cos(y) + I Inf sin(y)
	 */
	if (ix == 0x7ff00000 && lx == 0) {
		if (iy >= 0x7ff00000)
			return (CMPLX(x, y - y));
		return (CMPLX(x * cos(y), INFINITY * sin(y)));
	}

	/*
	 * sinh(NaN1 + I NaN2) = d(NaN1, NaN2) + I d(NaN1, NaN2).
	 *
	 * sinh(NaN +- I Inf)  = d(NaN, dNaN) + I d(NaN, dNaN).
	 * Optionally raises the invalid floating-point exception.
	 * Choice = raise.
	 *
	 * sinh(NaN + I y)     = d(NaN) + I d(NaN).
	 * Optionally raises the invalid floating-point exception for finite
	 * nonzero y.  Choice = don't raise (except for signaling NaNs).
	 */
	return (CMPLX(((long double)x + x) * (y - y),
	    ((long double)x * x) * (y - y)));
}

double complex
csin(double complex z)
{

	/* csin(z) = -I * csinh(I * z) = I * conj(csinh(I * conj(z))). */
	z = csinh(CMPLX(cimag(z), creal(z)));
	return (CMPLX(cimag(z), creal(z)));
}
