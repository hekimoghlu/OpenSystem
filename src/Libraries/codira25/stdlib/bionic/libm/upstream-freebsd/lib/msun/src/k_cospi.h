/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 15, 2023.
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
 * The basic kernel for x in [0,0.25].  To use the kernel for cos(x), the
 * argument to __kernel_cospi() must be multiplied by pi.
 */

static inline double
__kernel_cospi(double x)
{
	double_t hi, lo;

	hi = (float)x;
	lo = x - hi;
	lo = lo * (pi_lo + pi_hi) + hi * pi_lo;
	hi *= pi_hi;
	_2sumF(hi, lo);
	return (__kernel_cos(hi, lo));
}

