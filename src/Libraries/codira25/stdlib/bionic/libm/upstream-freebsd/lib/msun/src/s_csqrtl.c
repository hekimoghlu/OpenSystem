/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 4, 2023.
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

/*
 * Several thresholds require a 15-bit exponent and also the usual bias.
 * s_logl.c and e_hypotl have less hard-coding but end up requiring the
 * same for the exponent and more for the mantissa.
 */
#if LDBL_MAX_EXP != 0x4000
#error "Unsupported long double format"
#endif

/*
 * Overflow must be avoided for components >= LDBL_MAX / (1 + sqrt(2)).
 * The precise threshold is nontrivial to determine and spell, so use a
 * lower threshold of approximaely LDBL_MAX / 4, and don't use LDBL_MAX
 * to spell this since LDBL_MAX is broken on i386 (it overflows in 53-bit
 * precision).
 */
#define	THRESH	0x1p16382L

long double complex
csqrtl(long double complex z)
{
	long double complex result;
	long double a, b, rx, ry, scale, t;

	a = creall(z);
	b = cimagl(z);

	/* Handle special cases. */
	if (z == 0)
		return (CMPLXL(0, b));
	if (isinf(b))
		return (CMPLXL(INFINITY, b));
	if (isnan(a)) {
		t = (b - b) / (b - b);	/* raise invalid if b is not a NaN */
		return (CMPLXL(a + 0.0L + t, a + 0.0L + t)); /* NaN + NaN i */
	}
	if (isinf(a)) {
		/*
		 * csqrt(inf + NaN i)  = inf +  NaN i
		 * csqrt(inf + y i)    = inf +  0 i
		 * csqrt(-inf + NaN i) = NaN +- inf i
		 * csqrt(-inf + y i)   = 0   +  inf i
		 */
		if (signbit(a))
			return (CMPLXL(fabsl(b - b), copysignl(a, b)));
		else
			return (CMPLXL(a, copysignl(b - b, b)));
	}
	if (isnan(b)) {
		t = (a - a) / (a - a);	/* raise invalid */
		return (CMPLXL(b + 0.0L + t, b + 0.0L + t)); /* NaN + NaN i */
	}

	/* Scale to avoid overflow. */
	if (fabsl(a) >= THRESH || fabsl(b) >= THRESH) {
		/*
		 * Don't scale a or b if this might give (spurious)
		 * underflow.  Then the unscaled value is an equivalent
		 * infinitesmal (or 0).
		 */
		if (fabsl(a) >= 0x1p-16380L)
			a *= 0.25;
		if (fabsl(b) >= 0x1p-16380L)
			b *= 0.25;
		scale = 2;
	} else {
		scale = 1;
	}

	/* Scale to reduce inaccuracies when both components are denormal. */
	if (fabsl(a) < 0x1p-16382L && fabsl(b) < 0x1p-16382L) {
		a *= 0x1p64;
		b *= 0x1p64;
		scale = 0x1p-32;
	}

	/* Algorithm 312, CACM vol 10, Oct 1967. */
	if (a >= 0) {
		t = sqrtl((a + hypotl(a, b)) * 0.5);
		rx = scale * t;
		ry = scale * b / (2 * t);
	} else {
		t = sqrtl((-a + hypotl(a, b)) * 0.5);
		rx = scale * fabsl(b) / (2 * t);
		ry = copysignl(scale * t, b);
	}

	return (CMPLXL(rx, ry));
}
