/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 6, 2024.
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
 * truncl(x)
 * Return x rounded toward 0 to integral value
 * Method:
 *	Bit twiddling.
 * Exception:
 *	Inexact flag raised if x not equal to truncl(x).
 */

#include <float.h>
#include <math.h>
#include <stdint.h>

#include "fpmath.h"

#ifdef LDBL_IMPLICIT_NBIT
#define	MANH_SIZE	(LDBL_MANH_SIZE + 1)
#else
#define	MANH_SIZE	LDBL_MANH_SIZE
#endif

static const long double huge = 1.0e300;
static const float zero[] = { 0.0, -0.0 };

long double
truncl(long double x)
{
	union IEEEl2bits u = { .e = x };
	int e = u.bits.exp - LDBL_MAX_EXP + 1;

	if (e < MANH_SIZE - 1) {
		if (e < 0) {			/* raise inexact if x != 0 */
			if (huge + x > 0.0)
				u.e = zero[u.bits.sign];
		} else {
			uint64_t m = ((1llu << MANH_SIZE) - 1) >> (e + 1);
			if (((u.bits.manh & m) | u.bits.manl) == 0)
				return (x);	/* x is integral */
			if (huge + x > 0.0) {	/* raise inexact flag */
				u.bits.manh &= ~m;
				u.bits.manl = 0;
			}
		}
	} else if (e < LDBL_MANT_DIG - 1) {
		uint64_t m = (uint64_t)-1 >> (64 - LDBL_MANT_DIG + e + 1);
		if ((u.bits.manl & m) == 0)
			return (x);	/* x is integral */
		if (huge + x > 0.0)		/* raise inexact flag */
			u.bits.manl &= ~m;
	}
	return (u.e);
}
