/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 23, 2023.
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
one =  1.0000000000e+00, /* 0x3F800000 */
ln2 =  6.9314718246e-01, /* 0x3f317218 */
huge=  1.0000000000e+30;

float
asinhf(float x)
{
	float t,w;
	int32_t hx,ix;
	GET_FLOAT_WORD(hx,x);
	ix = hx&0x7fffffff;
	if(ix>=0x7f800000) return x+x;	/* x is inf or NaN */
	if(ix< 0x31800000) {	/* |x|<2**-28 */
	    if(huge+x>one) return x;	/* return x inexact except 0 */
	}
	if(ix>0x4d800000) {	/* |x| > 2**28 */
	    w = logf(fabsf(x))+ln2;
	} else if (ix>0x40000000) {	/* 2**28 > |x| > 2.0 */
	    t = fabsf(x);
	    w = logf((float)2.0*t+one/(sqrtf(x*x+one)+t));
	} else {		/* 2.0 > |x| > 2**-28 */
	    t = x*x;
	    w =log1pf(fabsf(x)+t/(one+sqrtf(one+t)));
	}
	if(hx>0) return w; else return -w;
}
