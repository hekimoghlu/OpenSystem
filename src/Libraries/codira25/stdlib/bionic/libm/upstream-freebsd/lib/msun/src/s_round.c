/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 6, 2025.
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

#include "math.h"
#include "math_private.h"

double
round(double x)
{
	double t;
	uint32_t hx;

	GET_HIGH_WORD(hx, x);
	if ((hx & 0x7fffffff) == 0x7ff00000)
		return (x + x);

	if (!(hx & 0x80000000)) {
		t = floor(x);
		if (t - x <= -0.5)
			t += 1;
		return (t);
	} else {
		t = floor(-x);
		if (t + x <= -0.5)
			t += 1;
		return (-t);
	}
}

#if (LDBL_MANT_DIG == 53)
__weak_reference(round, roundl);
#endif
