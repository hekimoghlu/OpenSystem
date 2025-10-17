/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 17, 2022.
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
 * Float version of ccosh().  See s_ccosh.c for details.
 */

#include <complex.h>
#include <math.h>

#include "math_private.h"

static const float huge = 0x1p127;

float complex
ccoshf(float complex z)
{
	float x, y, h;
	int32_t hx, hy, ix, iy;

	x = crealf(z);
	y = cimagf(z);

	GET_FLOAT_WORD(hx, x);
	GET_FLOAT_WORD(hy, y);

	ix = 0x7fffffff & hx;
	iy = 0x7fffffff & hy;

	if (ix < 0x7f800000 && iy < 0x7f800000) {
		if (iy == 0)
			return (CMPLXF(coshf(x), x * y));
		if (ix < 0x41100000)	/* |x| < 9: normal case */
			return (CMPLXF(coshf(x) * cosf(y), sinhf(x) * sinf(y)));

		/* |x| >= 9, so cosh(x) ~= exp(|x|) */
		if (ix < 0x42b17218) {
			/* x < 88.7: expf(|x|) won't overflow */
			h = expf(fabsf(x)) * 0.5F;
			return (CMPLXF(h * cosf(y), copysignf(h, x) * sinf(y)));
		} else if (ix < 0x4340b1e7) {
			/* x < 192.7: scale to avoid overflow */
			z = __ldexp_cexpf(CMPLXF(fabsf(x), y), -1);
			return (CMPLXF(crealf(z), cimagf(z) * copysignf(1, x)));
		} else {
			/* x >= 192.7: the result always overflows */
			h = huge * x;
			return (CMPLXF(h * h * cosf(y), h * sinf(y)));
		}
	}

	if (ix == 0)			/* && iy >= 0x7f800000 */
		return (CMPLXF(y - y, x * copysignf(0, y)));

	if (iy == 0)			/* && ix >= 0x7f800000 */
		return (CMPLXF(x * x, copysignf(0, x) * y));

	if (ix < 0x7f800000)		/* && iy >= 0x7f800000 */
		return (CMPLXF(y - y, x * (y - y)));

	if (ix == 0x7f800000) {
		if (iy >= 0x7f800000)
			return (CMPLXF(INFINITY, x * (y - y)));
		return (CMPLXF(INFINITY * cosf(y), x * sinf(y)));
	}

	return (CMPLXF(((long double)x * x) * (y - y),
	    ((long double)x + x) * (y - y)));
}

float complex
ccosf(float complex z)
{

	return (ccoshf(CMPLXF(-cimagf(z), crealf(z))));
}
