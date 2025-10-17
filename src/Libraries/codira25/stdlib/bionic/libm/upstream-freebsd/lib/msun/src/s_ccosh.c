/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 17, 2025.
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
 * Hyperbolic cosine of a complex argument z = x + i y.
 *
 * cosh(z) = cosh(x+iy)
 *         = cosh(x) cos(y) + i sinh(x) sin(y).
 *
 * Exceptional values are noted in the comments within the source code.
 * These values and the return value were taken from n1124.pdf.
 * The sign of the result for some exceptional values is unspecified but
 * must satisfy both cosh(conj(z)) == conj(cosh(z)) and cosh(-z) == cosh(z).
 */

#include <complex.h>
#include <math.h>

#include "math_private.h"

static const double huge = 0x1p1023;

double complex
ccosh(double complex z)
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
			return (CMPLX(cosh(x), x * y));
		if (ix < 0x40360000)	/* |x| < 22: normal case */
			return (CMPLX(cosh(x) * cos(y), sinh(x) * sin(y)));

		/* |x| >= 22, so cosh(x) ~= exp(|x|) */
		if (ix < 0x40862e42) {
			/* x < 710: exp(|x|) won't overflow */
			h = exp(fabs(x)) * 0.5;
			return (CMPLX(h * cos(y), copysign(h, x) * sin(y)));
		} else if (ix < 0x4096bbaa) {
			/* x < 1455: scale to avoid overflow */
			z = __ldexp_cexp(CMPLX(fabs(x), y), -1);
			return (CMPLX(creal(z), cimag(z) * copysign(1, x)));
		} else {
			/* x >= 1455: the result always overflows */
			h = huge * x;
			return (CMPLX(h * h * cos(y), h * sin(y)));
		}
	}

	/*
	 * cosh(+-0 +- I Inf) = dNaN + I (+-)(+-)0.
	 * The sign of 0 in the result is unspecified.  Choice = product
	 * of the signs of the argument.  Raise the invalid floating-point
	 * exception.
	 *
	 * cosh(+-0 +- I NaN) = d(NaN) + I (+-)(+-)0.
	 * The sign of 0 in the result is unspecified.  Choice = product
	 * of the signs of the argument.
	 */
	if ((ix | lx) == 0)		/* && iy >= 0x7ff00000 */
		return (CMPLX(y - y, x * copysign(0, y)));

	/*
	 * cosh(+-Inf +- I 0) = +Inf + I (+-)(+-)0.
	 *
	 * cosh(NaN +- I 0)   = d(NaN) + I (+-)(+-)0.
	 * The sign of 0 in the result is unspecified.  Choice = product
	 * of the signs of the argument.
	 */
	if ((iy | ly) == 0)		/* && ix >= 0x7ff00000 */
		return (CMPLX(x * x, copysign(0, x) * y));

	/*
	 * cosh(x +- I Inf) = dNaN + I dNaN.
	 * Raise the invalid floating-point exception for finite nonzero x.
	 *
	 * cosh(x + I NaN) = d(NaN) + I d(NaN).
	 * Optionally raises the invalid floating-point exception for finite
	 * nonzero x.  Choice = don't raise (except for signaling NaNs).
	 */
	if (ix < 0x7ff00000)		/* && iy >= 0x7ff00000 */
		return (CMPLX(y - y, x * (y - y)));

	/*
	 * cosh(+-Inf + I NaN)  = +Inf + I d(NaN).
	 *
	 * cosh(+-Inf +- I Inf) = +Inf + I dNaN.
	 * The sign of Inf in the result is unspecified.  Choice = always +.
	 * Raise the invalid floating-point exception.
	 *
	 * cosh(+-Inf + I y)   = +Inf cos(y) +- I Inf sin(y)
	 */
	if (ix == 0x7ff00000 && lx == 0) {
		if (iy >= 0x7ff00000)
			return (CMPLX(INFINITY, x * (y - y)));
		return (CMPLX(INFINITY * cos(y), x * sin(y)));
	}

	/*
	 * cosh(NaN + I NaN)  = d(NaN) + I d(NaN).
	 *
	 * cosh(NaN +- I Inf) = d(NaN) + I d(NaN).
	 * Optionally raises the invalid floating-point exception.
	 * Choice = raise.
	 *
	 * cosh(NaN + I y)    = d(NaN) + I d(NaN).
	 * Optionally raises the invalid floating-point exception for finite
	 * nonzero y.  Choice = don't raise (except for signaling NaNs).
	 */
	return (CMPLX(((long double)x * x) * (y - y),
	    ((long double)x + x) * (y - y)));
}

double complex
ccos(double complex z)
{

	/* ccos(z) = ccosh(I * z) */
	return (ccosh(CMPLX(-cimag(z), creal(z))));
}
