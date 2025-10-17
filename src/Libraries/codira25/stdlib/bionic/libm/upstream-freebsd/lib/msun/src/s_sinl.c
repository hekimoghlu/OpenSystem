/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 2, 2023.
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
#if LDBL_MANT_DIG == 64
#include "../ld80/e_rem_pio2l.h"
#elif LDBL_MANT_DIG == 113
#include "../ld128/e_rem_pio2l.h"
#else
#error "Unsupported long double format"
#endif

long double
sinl(long double x)
{
	union IEEEl2bits z;
	int e0, s;
	long double y[2];
	long double hi, lo;

	z.e = x;
	s = z.bits.sign;
	z.bits.sign = 0;

	/* If x = +-0 or x is a subnormal number, then sin(x) = x */
	if (z.bits.exp == 0)
		return (x);

	/* If x = NaN or Inf, then sin(x) = NaN. */
	if (z.bits.exp == 32767)
		return ((x - x) / (x - x));

	ENTERI();

	/* Optimize the case where x is already within range. */
	if (z.e < M_PI_4) {
		hi = __kernel_sinl(z.e, 0, 0);
		RETURNI(s ? -hi : hi);
	}

	e0 = __ieee754_rem_pio2l(x, y);
	hi = y[0];
	lo = y[1];

	switch (e0 & 3) {
	case 0:
	    hi = __kernel_sinl(hi, lo, 1);
	    break;
	case 1:
	    hi = __kernel_cosl(hi, lo);
	    break;
	case 2:
	    hi = - __kernel_sinl(hi, lo, 1);
	    break;
	case 3:
	    hi = - __kernel_cosl(hi, lo);
	    break;
	}
	
	RETURNI(hi);
}
