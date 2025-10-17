/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 23, 2022.
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

#include "fpmath.h"

#ifdef USE_BUILTIN_FMINF
float
fminf(float x, float y)
{
	return (__builtin_fminf(x, y));
}
#else
float
fminf(float x, float y)
{
	union IEEEf2bits u[2];

	u[0].f = x;
	u[1].f = y;

	/* Check for NaNs to avoid raising spurious exceptions. */
	if (u[0].bits.exp == 255 && u[0].bits.man != 0)
		return (y);
	if (u[1].bits.exp == 255 && u[1].bits.man != 0)
		return (x);

	/* Handle comparisons of signed zeroes. */
	if (u[0].bits.sign != u[1].bits.sign)
		return (u[u[1].bits.sign].f);

	return (x < y ? x : y);
}
#endif
