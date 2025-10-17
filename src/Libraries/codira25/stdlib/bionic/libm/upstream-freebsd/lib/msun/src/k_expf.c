/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 3, 2023.
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

#include "math.h"
#include "math_private.h"

static const uint32_t k = 235;			/* constant for reduction */
static const float kln2 =  162.88958740F;	/* k * ln2 */

/*
 * See k_exp.c for details.
 *
 * Input:  ln(FLT_MAX) <= x < ln(2 * FLT_MAX / FLT_MIN_DENORM) ~= 192.7
 * Output: 2**127 <= y < 2**128
 */
static float
__frexp_expf(float x, int *expt)
{
	float exp_x;
	uint32_t hx;

	exp_x = expf(x - kln2);
	GET_FLOAT_WORD(hx, exp_x);
	*expt = (hx >> 23) - (0x7f + 127) + k;
	SET_FLOAT_WORD(exp_x, (hx & 0x7fffff) | ((0x7f + 127) << 23));
	return (exp_x);
}

float
__ldexp_expf(float x, int expt)
{
	float exp_x, scale;
	int ex_expt;

	exp_x = __frexp_expf(x, &ex_expt);
	expt += ex_expt;
	SET_FLOAT_WORD(scale, (0x7f + expt) << 23);
	return (exp_x * scale);
}

float complex
__ldexp_cexpf(float complex z, int expt)
{
	float c, exp_x, s, scale1, scale2, x, y;
	int ex_expt, half_expt;

	x = crealf(z);
	y = cimagf(z);
	exp_x = __frexp_expf(x, &ex_expt);
	expt += ex_expt;

	half_expt = expt / 2;
	SET_FLOAT_WORD(scale1, (0x7f + half_expt) << 23);
	half_expt = expt - half_expt;
	SET_FLOAT_WORD(scale2, (0x7f + half_expt) << 23);

	sincosf(y, &s, &c);
	return (CMPLXF(c * exp_x * scale1 * scale2,
	    s * exp_x * scale1 * scale2));
}
