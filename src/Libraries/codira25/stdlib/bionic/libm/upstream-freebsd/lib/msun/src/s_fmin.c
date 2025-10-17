/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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
#include <math.h>

#include "fpmath.h"

#ifdef USE_BUILTIN_FMIN
double
fmin(double x, double y)
{
	return (__builtin_fmin(x, y));
}
#else
double
fmin(double x, double y)
{
	union IEEEd2bits u[2];

	u[0].d = x;
	u[1].d = y;

	/* Check for NaNs to avoid raising spurious exceptions. */
	if (u[0].bits.exp == 2047 && (u[0].bits.manh | u[0].bits.manl) != 0)
		return (y);
	if (u[1].bits.exp == 2047 && (u[1].bits.manh | u[1].bits.manl) != 0)
		return (x);

	/* Handle comparisons of signed zeroes. */
	if (u[0].bits.sign != u[1].bits.sign)
		return (u[u[1].bits.sign].d);

	return (x < y ? x : y);
}
#endif

#if (LDBL_MANT_DIG == 53)
__weak_reference(fmin, fminl);
#endif
