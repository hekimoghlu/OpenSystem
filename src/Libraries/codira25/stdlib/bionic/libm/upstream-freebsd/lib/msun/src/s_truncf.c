/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 5, 2024.
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
 * truncf(x)
 * Return x rounded toward 0 to integral value
 * Method:
 *	Bit twiddling.
 * Exception:
 *	Inexact flag raised if x not equal to truncf(x).
 */

#include "math.h"
#include "math_private.h"

static const float huge = 1.0e30F;

float
truncf(float x)
{
	int32_t i0,j0;
	u_int32_t i;
	GET_FLOAT_WORD(i0,x);
	j0 = ((i0>>23)&0xff)-0x7f;
	if(j0<23) {
	    if(j0<0) { 	/* raise inexact if x != 0 */
		if(huge+x>0.0F)		/* |x|<1, so return 0*sign(x) */
		    i0 &= 0x80000000;
	    } else {
		i = (0x007fffff)>>j0;
		if((i0&i)==0) return x; /* x is integral */
		if(huge+x>0.0F)		/* raise inexact flag */
		    i0 &= (~i);
	    }
	} else {
	    if(j0==0x80) return x+x;	/* inf or NaN */
	    else return x;		/* x is integral */
	}
	SET_FLOAT_WORD(x,i0);
	return x;
}
