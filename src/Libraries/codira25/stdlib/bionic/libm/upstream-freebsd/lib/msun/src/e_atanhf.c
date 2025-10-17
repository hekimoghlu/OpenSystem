/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 21, 2025.
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

static const float one = 1.0, huge = 1e30;

static const float zero = 0.0;

float
atanhf(float x)
{
	float t;
	int32_t hx,ix;
	GET_FLOAT_WORD(hx,x);
	ix = hx&0x7fffffff;
	if (ix>0x3f800000) 		/* |x|>1 */
	    return (x-x)/(x-x);
	if(ix==0x3f800000)
	    return x/zero;
	if(ix<0x31800000&&(huge+x)>zero) return x;	/* x<2**-28 */
	SET_FLOAT_WORD(x,ix);
	if(ix<0x3f000000) {		/* x < 0.5 */
	    t = x+x;
	    t = (float)0.5*log1pf(t+t*x/(one-x));
	} else
	    t = (float)0.5*log1pf((x+x)/(one-x));
	if(hx>=0) return t; else return -t;
}
