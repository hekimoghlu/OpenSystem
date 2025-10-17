/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 12, 2023.
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

#include "fpmath.h"
#include "math.h"
#include "math_private.h"

#define	MANT_DIG	FLT_MANT_DIG
#define	MAX_EXP		FLT_MAX_EXP
#define	MIN_EXP		FLT_MIN_EXP

static const float
ln2f_hi =  6.9314575195e-1,		/*  0xb17200.0p-24 */
ln2f_lo =  1.4286067653e-6;		/*  0xbfbe8e.0p-43 */

float complex
clogf(float complex z)
{
	float_t ax, ax2h, ax2l, axh, axl, ay, ay2h, ay2l, ayh, ayl, sh, sl, t;
	float x, y, v;
	uint32_t hax, hay;
	int kx, ky;

	x = crealf(z);
	y = cimagf(z);
	v = atan2f(y, x);

	ax = fabsf(x);
	ay = fabsf(y);
	if (ax < ay) {
		t = ax;
		ax = ay;
		ay = t;
	}

	GET_FLOAT_WORD(hax, ax);
	kx = (hax >> 23) - 127;
	GET_FLOAT_WORD(hay, ay);
	ky = (hay >> 23) - 127;

	/* Handle NaNs and Infs using the general formula. */
	if (kx == MAX_EXP || ky == MAX_EXP)
		return (CMPLXF(logf(hypotf(x, y)), v));

	/* Avoid spurious underflow, and reduce inaccuracies when ax is 1. */
	if (hax == 0x3f800000) {
		if (ky < (MIN_EXP - 1) / 2)
			return (CMPLXF((ay / 2) * ay, v));
		return (CMPLXF(log1pf(ay * ay) / 2, v));
	}

	/* Avoid underflow when ax is not small.  Also handle zero args. */
	if (kx - ky > MANT_DIG || hay == 0)
		return (CMPLXF(logf(ax), v));

	/* Avoid overflow. */
	if (kx >= MAX_EXP - 1)
		return (CMPLXF(logf(hypotf(x * 0x1p-126F, y * 0x1p-126F)) +
		    (MAX_EXP - 2) * ln2f_lo + (MAX_EXP - 2) * ln2f_hi, v));
	if (kx >= (MAX_EXP - 1) / 2)
		return (CMPLXF(logf(hypotf(x, y)), v));

	/* Reduce inaccuracies and avoid underflow when ax is denormal. */
	if (kx <= MIN_EXP - 2)
		return (CMPLXF(logf(hypotf(x * 0x1p127F, y * 0x1p127F)) +
		    (MIN_EXP - 2) * ln2f_lo + (MIN_EXP - 2) * ln2f_hi, v));

	/* Avoid remaining underflows (when ax is small but not denormal). */
	if (ky < (MIN_EXP - 1) / 2 + MANT_DIG)
		return (CMPLXF(logf(hypotf(x, y)), v));

	/* Calculate ax*ax and ay*ay exactly using Dekker's algorithm. */
	t = (float)(ax * (0x1p12F + 1));
	axh = (float)(ax - t) + t;
	axl = ax - axh;
	ax2h = ax * ax;
	ax2l = axh * axh - ax2h + 2 * axh * axl + axl * axl;
	t = (float)(ay * (0x1p12F + 1));
	ayh = (float)(ay - t) + t;
	ayl = ay - ayh;
	ay2h = ay * ay;
	ay2l = ayh * ayh - ay2h + 2 * ayh * ayl + ayl * ayl;

	/*
	 * When log(|z|) is far from 1, accuracy in calculating the sum
	 * of the squares is not very important since log() reduces
	 * inaccuracies.  We depended on this to use the general
	 * formula when log(|z|) is very far from 1.  When log(|z|) is
	 * moderately far from 1, we go through the extra-precision
	 * calculations to reduce branches and gain a little accuracy.
	 *
	 * When |z| is near 1, we subtract 1 and use log1p() and don't
	 * leave it to log() to subtract 1, since we gain at least 1 bit
	 * of accuracy in this way.
	 *
	 * When |z| is very near 1, subtracting 1 can cancel almost
	 * 3*MANT_DIG bits.  We arrange that subtracting 1 is exact in
	 * doubled precision, and then do the rest of the calculation
	 * in sloppy doubled precision.  Although large cancellations
	 * often lose lots of accuracy, here the final result is exact
	 * in doubled precision if the large calculation occurs (because
	 * then it is exact in tripled precision and the cancellation
	 * removes enough bits to fit in doubled precision).  Thus the
	 * result is accurate in sloppy doubled precision, and the only
	 * significant loss of accuracy is when it is summed and passed
	 * to log1p().
	 */
	sh = ax2h;
	sl = ay2h;
	_2sumF(sh, sl);
	if (sh < 0.5F || sh >= 3)
		return (CMPLXF(logf(ay2l + ax2l + sl + sh) / 2, v));
	sh -= 1;
	_2sum(sh, sl);
	_2sum(ax2l, ay2l);
	/* Briggs-Kahan algorithm (except we discard the final low term): */
	_2sum(sh, ax2l);
	_2sum(sl, ay2l);
	t = ax2l + sl;
	_2sumF(sh, t);
	return (CMPLXF(log1pf(ay2l + t + sh) / 2, v));
}
