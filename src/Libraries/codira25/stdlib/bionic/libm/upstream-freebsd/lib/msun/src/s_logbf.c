/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 11, 2024.
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
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

#include "math.h"
#include "math_private.h"

static const float
two25 = 3.355443200e+07;		/* 0x4c000000 */

float
logbf(float x)
{
	int32_t ix;
	GET_FLOAT_WORD(ix,x);
	ix &= 0x7fffffff;			/* high |x| */
	if(ix==0) return (float)-1.0/fabsf(x);
	if(ix>=0x7f800000) return x*x;
	if(ix<0x00800000) {
		x *= two25;		 /* convert subnormal x to normal */
		GET_FLOAT_WORD(ix,x);
		ix &= 0x7fffffff;
		return (float) ((ix>>23)-127-25);
	} else
		return (float) ((ix>>23)-127);
}
