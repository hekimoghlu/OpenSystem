/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 7, 2025.
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

long double
fminl(long double x, long double y)
{
	union IEEEl2bits u[2];

	u[0].e = x;
	mask_nbit_l(u[0]);
	u[1].e = y;
	mask_nbit_l(u[1]);

	/* Check for NaNs to avoid raising spurious exceptions. */
	if (u[0].bits.exp == 32767 && (u[0].bits.manh | u[0].bits.manl) != 0)
		return (y);
	if (u[1].bits.exp == 32767 && (u[1].bits.manh | u[1].bits.manl) != 0)
		return (x);

	/* Handle comparisons of signed zeroes. */
	if (u[0].bits.sign != u[1].bits.sign)
		return (u[1].bits.sign ? y : x);

	return (x < y ? x : y);
}
