/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 17, 2022.
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
#include <math.h>

#include "math_private.h"

/* For avoiding overflow for components >= DBL_MAX / (1 + sqrt(2)). */
#define	THRESH	0x1.a827999fcef32p+1022

double complex
csqrt(double complex z)
{
	double complex result;
	double a, b, rx, ry, scale, t;

	a = creal(z);
	b = cimag(z);

	/* Handle special cases. */
	if (z == 0)
		return (CMPLX(0, b));
	if (isinf(b))
		return (CMPLX(INFINITY, b));
	if (isnan(a)) {
		t = (b - b) / (b - b);	/* raise invalid if b is not a NaN */
		return (CMPLX(a + 0.0L + t, a + 0.0L + t)); /* NaN + NaN i */
	}
	if (isinf(a)) {
		/*
		 * csqrt(inf + NaN i)  = inf +  NaN i
		 * csqrt(inf + y i)    = inf +  0 i
		 * csqrt(-inf + NaN i) = NaN +- inf i
		 * csqrt(-inf + y i)   = 0   +  inf i
		 */
		if (signbit(a))
			return (CMPLX(fabs(b - b), copysign(a, b)));
		else
			return (CMPLX(a, copysign(b - b, b)));
	}
	if (isnan(b)) {
		t = (a - a) / (a - a);	/* raise invalid */
		return (CMPLX(b + 0.0L + t, b + 0.0L + t)); /* NaN + NaN i */
	}

	/* Scale to avoid overflow. */
	if (fabs(a) >= THRESH || fabs(b) >= THRESH) {
		/*
		 * Don't scale a or b if this might give (spurious)
		 * underflow.  Then the unscaled value is an equivalent
		 * infinitesmal (or 0).
		 */
		if (fabs(a) >= 0x1p-1020)
			a *= 0.25;
		if (fabs(b) >= 0x1p-1020)
			b *= 0.25;
		scale = 2;
	} else {
		scale = 1;
	}

	/* Scale to reduce inaccuracies when both components are denormal. */
	if (fabs(a) < 0x1p-1022 && fabs(b) < 0x1p-1022) {
		a *= 0x1p54;
		b *= 0x1p54;
		scale = 0x1p-27;
	}

	/* Algorithm 312, CACM vol 10, Oct 1967. */
	if (a >= 0) {
		t = sqrt((a + hypot(a, b)) * 0.5);
		rx = scale * t;
		ry = scale * b / (2 * t);
	} else {
		t = sqrt((-a + hypot(a, b)) * 0.5);
		rx = scale * fabs(b) / (2 * t);
		ry = copysign(scale * t, b);
	}

	return (CMPLX(rx, ry));
}

#if LDBL_MANT_DIG == 53
__weak_reference(csqrt, csqrtl);
#endif
