/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 13, 2024.
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
 * Hyperbolic tangent of a complex argument z.  See s_ctanh.c for details.
 */

#include <complex.h>
#include <math.h>

#include "math_private.h"

float complex
ctanhf(float complex z)
{
	float x, y;
	float t, beta, s, rho, denom;
	uint32_t hx, ix;

	x = crealf(z);
	y = cimagf(z);

	GET_FLOAT_WORD(hx, x);
	ix = hx & 0x7fffffff;

	if (ix >= 0x7f800000) {
		if (ix & 0x7fffff)
			return (CMPLXF(nan_mix(x, y),
			    y == 0 ? y : nan_mix(x, y)));
		SET_FLOAT_WORD(x, hx - 0x40000000);
		return (CMPLXF(x,
		    copysignf(0, isinf(y) ? y : sinf(y) * cosf(y))));
	}

	if (!isfinite(y))
		return (CMPLXF(ix ? y - y : x, y - y));

	if (ix >= 0x41300000) {	/* |x| >= 11 */
		float exp_mx = expf(-fabsf(x));
		return (CMPLXF(copysignf(1, x),
		    4 * sinf(y) * cosf(y) * exp_mx * exp_mx));
	}

	t = tanf(y);
	beta = 1.0 + t * t;
	s = sinhf(x);
	rho = sqrtf(1 + s * s);
	denom = 1 + beta * s * s;
	return (CMPLXF((beta * rho * s) / denom, t / denom));
}

float complex
ctanf(float complex z)
{

	z = ctanhf(CMPLXF(cimagf(z), crealf(z)));
	return (CMPLXF(cimagf(z), crealf(z)));
}

