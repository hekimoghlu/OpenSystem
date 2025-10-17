/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 22, 2022.
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

#include <float.h>

#include "math.h"
#define	INLINE_KERNEL_TANDF
#define INLINE_REM_PIO2F
#include "math_private.h"
#include "e_rem_pio2f.c"
#include "k_tanf.c"

/* Small multiples of pi/2 rounded to double precision. */
static const double
t1pio2 = 1*M_PI_2,			/* 0x3FF921FB, 0x54442D18 */
t2pio2 = 2*M_PI_2,			/* 0x400921FB, 0x54442D18 */
t3pio2 = 3*M_PI_2,			/* 0x4012D97C, 0x7F3321D2 */
t4pio2 = 4*M_PI_2;			/* 0x401921FB, 0x54442D18 */

float
tanf(float x)
{
	double y;
	int32_t n, hx, ix;

	GET_FLOAT_WORD(hx,x);
	ix = hx & 0x7fffffff;

	if(ix <= 0x3f490fda) {		/* |x| ~<= pi/4 */
	    if(ix<0x39800000)		/* |x| < 2**-12 */
		if(((int)x)==0) return x;	/* x with inexact if x != 0 */
	    return __kernel_tandf(x,1);
	}
	if(ix<=0x407b53d1) {		/* |x| ~<= 5*pi/4 */
	    if(ix<=0x4016cbe3)		/* |x| ~<= 3pi/4 */
		return __kernel_tandf(x + (hx>0 ? -t1pio2 : t1pio2), -1);
	    else
		return __kernel_tandf(x + (hx>0 ? -t2pio2 : t2pio2), 1);
	}
	if(ix<=0x40e231d5) {		/* |x| ~<= 9*pi/4 */
	    if(ix<=0x40afeddf)		/* |x| ~<= 7*pi/4 */
		return __kernel_tandf(x + (hx>0 ? -t3pio2 : t3pio2), -1);
	    else
		return __kernel_tandf(x + (hx>0 ? -t4pio2 : t4pio2), 1);
	}

    /* tan(Inf or NaN) is NaN */
	else if (ix>=0x7f800000) return x-x;

    /* general argument reduction needed */
	else {
	    n = __ieee754_rem_pio2f(x,&y);
	    /* integer parameter: 1 -- n even; -1 -- n odd */
	    return __kernel_tandf(y,1-((n&1)<<1));
	}
}
