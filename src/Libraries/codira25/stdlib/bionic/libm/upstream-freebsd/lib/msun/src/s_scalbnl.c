/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 16, 2022.
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
#include <math.h>
#include <float.h>
#include "math_private.h"
#include "fpmath.h"
/*
 * scalbnl (long double x, int n)
 * scalbnl(x,n) returns x* 2**n  computed by  exponent
 * manipulation rather than by actually performing an
 * exponentiation or a multiplication.
 */
#if (LDBL_MANT_DIG == 64 || LDBL_MANT_DIG == 113) && LDBL_MAX_EXP == 16384
long double scalbnl(long double x, int n)
{
	union IEEEl2bits u;

	if (n > 16383) {
		x *= 0x1p16383L;
		n -= 16383;
		if (n > 16383) {
			x *= 0x1p16383L;
			n -= 16383;
			if (n > 16383)
				n = 16383;
		}
	} else if (n < -16382) {
		x *= 0x1p-16382L * 0x1p113L;
		n += 16382 - 113;
		if (n < -16382) {
			x *= 0x1p-16382L * 0x1p113L;
			n += 16382 - 113;
			if (n < -16382)
				n = -16382;
		}
	}
	u.e = 1.0;
	u.xbits.expsign = 0x3fff + n;
	return x * u.e;
}
__strong_reference(scalbnl, ldexpl);
#endif

