/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 3, 2024.
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
#include <float.h>
#ifdef __i386__
#include <ieeefp.h>
#endif

#include "math.h"
#include "math_private.h"
#include "k_sincosl.h"

#if LDBL_MANT_DIG == 64
#include "../ld80/e_rem_pio2l.h"
#elif LDBL_MANT_DIG == 113
#include "../ld128/e_rem_pio2l.h"
#else
#error "Unsupported long double format"
#endif

void
sincosl(long double x, long double *sn, long double *cs)
{
	union IEEEl2bits z;
	int e0;
	long double y[2];

	z.e = x;
	z.bits.sign = 0;

	ENTERV();

	/* Optimize the case where x is already within range. */
	if (z.e < M_PI_4) {
		/*
		 * If x = +-0 or x is a subnormal number, then sin(x) = x and
		 * cos(x) = 1.
		 */
		if (z.bits.exp == 0) {
			*sn = x;
			*cs = 1;
		} else
			__kernel_sincosl(x, 0, 0, sn, cs);
		RETURNV();
	}

	/* If x = NaN or Inf, then sin(x) and cos(x) are NaN. */
	if (z.bits.exp == 32767) {
		*sn = x - x;
		*cs = x - x;
		RETURNV();
	}

	/* Range reduction. */
	e0 = __ieee754_rem_pio2l(x, y);

	switch (e0 & 3) {
	case 0:
		__kernel_sincosl(y[0], y[1], 1, sn, cs);
		break;
	case 1:
		__kernel_sincosl(y[0], y[1], 1, cs, sn);
		*cs = -*cs;
		break;
	case 2:
		__kernel_sincosl(y[0], y[1], 1, sn, cs);
		*sn = -*sn;
		*cs = -*cs;
		break;
	default:
		__kernel_sincosl(y[0], y[1], 1, cs, sn);
		*sn = -*sn;
	}

	RETURNV();
}
