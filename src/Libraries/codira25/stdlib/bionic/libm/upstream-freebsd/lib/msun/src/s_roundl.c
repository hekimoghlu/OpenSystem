/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 18, 2024.
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

#include "fpmath.h"
#include "math.h"
#include "math_private.h"

long double
roundl(long double x)
{
	long double t;
	uint16_t hx;

	GET_LDBL_EXPSIGN(hx, x);
	if ((hx & 0x7fff) == 0x7fff)
		return (x + x);

	ENTERI();

	if (!(hx & 0x8000)) {
		t = floorl(x);
		if (t - x <= -0.5L)
			t += 1;
		RETURNI(t);
	} else {
		t = floorl(-x);
		if (t + x <= -0.5L)
			t += 1;
		RETURNI(-t);
	}
}
