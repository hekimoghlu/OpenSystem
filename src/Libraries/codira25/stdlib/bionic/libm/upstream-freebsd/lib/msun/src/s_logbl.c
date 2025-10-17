/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 11, 2023.
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
#include <limits.h>
#include <math.h>

#include "fpmath.h"

long double
logbl(long double x)
{
	union IEEEl2bits u;
	unsigned long m;
	int b;

	u.e = x;
	if (u.bits.exp == 0) {
		if ((u.bits.manl | u.bits.manh) == 0) {	/* x == 0 */
			u.bits.sign = 1;
			return (1.0L / u.e);
		}
		/* denormalized */
		if (u.bits.manh == 0) {
			m = 1lu << (LDBL_MANL_SIZE - 1);
			for (b = LDBL_MANH_SIZE; !(u.bits.manl & m); m >>= 1)
				b++;
		} else {
			m = 1lu << (LDBL_MANH_SIZE - 1);
			for (b = 0; !(u.bits.manh & m); m >>= 1)
				b++;
		}
#ifdef LDBL_IMPLICIT_NBIT
		b++;
#endif
		return ((long double)(LDBL_MIN_EXP - b - 1));
	}
	if (u.bits.exp < (LDBL_MAX_EXP << 1) - 1)	/* normal */
		return ((long double)(u.bits.exp - LDBL_MAX_EXP + 1));
	else						/* +/- inf or nan */
		return (x * x);
}
