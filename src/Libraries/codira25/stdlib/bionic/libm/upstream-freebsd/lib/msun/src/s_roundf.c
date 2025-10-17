/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 23, 2025.
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
#include "math.h"
#include "math_private.h"

float
roundf(float x)
{
	float t;
	uint32_t hx;

	GET_FLOAT_WORD(hx, x);
	if ((hx & 0x7fffffff) == 0x7f800000)
		return (x + x);

	if (!(hx & 0x80000000)) {
		t = floorf(x);
		if (t - x <= -0.5F)
			t += 1;
		return (t);
	} else {
		t = floorf(-x);
		if (t + x <= -0.5F)
			t += 1;
		return (-t);
	}
}
