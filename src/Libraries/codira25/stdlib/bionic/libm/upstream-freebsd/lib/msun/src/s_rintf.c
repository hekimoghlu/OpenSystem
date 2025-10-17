/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 11, 2022.
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
#include <stdint.h>

#include "math.h"
#include "math_private.h"

static const float
TWO23[2]={
  8.3886080000e+06, /* 0x4b000000 */
 -8.3886080000e+06, /* 0xcb000000 */
};

float
rintf(float x)
{
	int32_t i0,j0,sx;
	float w,t;
	GET_FLOAT_WORD(i0,x);
	sx = (i0>>31)&1;
	j0 = ((i0>>23)&0xff)-0x7f;
	if(j0<23) {
	    if(j0<0) {
		if((i0&0x7fffffff)==0) return x;
		STRICT_ASSIGN(float,w,TWO23[sx]+x);
	        t =  w-TWO23[sx];
		GET_FLOAT_WORD(i0,t);
		SET_FLOAT_WORD(t,(i0&0x7fffffff)|(sx<<31));
	        return t;
	    }
	    STRICT_ASSIGN(float,w,TWO23[sx]+x);
	    return w-TWO23[sx];
	}
	if(j0==0x80) return x+x;	/* inf or NaN */
	else return x;			/* x is integral */
}
