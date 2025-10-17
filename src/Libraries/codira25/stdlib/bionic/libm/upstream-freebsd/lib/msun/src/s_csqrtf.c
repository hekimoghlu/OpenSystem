/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 30, 2024.
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
#include <math.h>

#include "math_private.h"

float complex
csqrtf(float complex z)
{
	double t;
	float a, b;

	a = creal(z);
	b = cimag(z);

	/* Handle special cases. */
	if (z == 0)
		return (CMPLXF(0, b));
	if (isinf(b))
		return (CMPLXF(INFINITY, b));
	if (isnan(a)) {
		t = (b - b) / (b - b);	/* raise invalid if b is not a NaN */
		return (CMPLXF(a + 0.0L + t, a + 0.0L + t)); /* NaN + NaN i */
	}
	if (isinf(a)) {
		/*
		 * csqrtf(inf + NaN i)  = inf +  NaN i
		 * csqrtf(inf + y i)    = inf +  0 i
		 * csqrtf(-inf + NaN i) = NaN +- inf i
		 * csqrtf(-inf + y i)   = 0   +  inf i
		 */
		if (signbit(a))
			return (CMPLXF(fabsf(b - b), copysignf(a, b)));
		else
			return (CMPLXF(a, copysignf(b - b, b)));
	}
	if (isnan(b)) {
		t = (a - a) / (a - a);	/* raise invalid */
		return (CMPLXF(b + 0.0L + t, b + 0.0L + t)); /* NaN + NaN i */
	}

	/*
	 * We compute t in double precision to avoid overflow and to
	 * provide correct rounding in nearly all cases.
	 * This is Algorithm 312, CACM vol 10, Oct 1967.
	 */
	if (a >= 0) {
		t = sqrt((a + hypot(a, b)) * 0.5);
		return (CMPLXF(t, b / (2 * t)));
	} else {
		t = sqrt((-a + hypot(a, b)) * 0.5);
		return (CMPLXF(fabsf(b) / (2 * t), copysignf(t, b)));
	}
}
