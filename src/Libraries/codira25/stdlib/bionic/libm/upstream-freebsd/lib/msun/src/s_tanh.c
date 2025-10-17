/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 3, 2024.
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
/* Tanh(x)
 * Return the Hyperbolic Tangent of x
 *
 * Method :
 *				       x    -x
 *				      e  - e
 *	0. tanh(x) is defined to be -----------
 *				       x    -x
 *				      e  + e
 *	1. reduce x to non-negative by tanh(-x) = -tanh(x).
 *	2.  0      <= x <  2**-28 : tanh(x) := x with inexact if x != 0
 *					        -t
 *	    2**-28 <= x <  1      : tanh(x) := -----; t = expm1(-2x)
 *					       t + 2
 *						     2
 *	    1      <= x <  22     : tanh(x) := 1 - -----; t = expm1(2x)
 *						   t + 2
 *	    22     <= x <= INF    : tanh(x) := 1.
 *
 * Special cases:
 *	tanh(NaN) is NaN;
 *	only tanh(0)=0 is exact for finite argument.
 */

#include <float.h>

#include "math.h"
#include "math_private.h"

static const volatile double tiny = 1.0e-300;
static const double one = 1.0, two = 2.0, huge = 1.0e300;

double
tanh(double x)
{
	double t,z;
	int32_t jx,ix;

	GET_HIGH_WORD(jx,x);
	ix = jx&0x7fffffff;

    /* x is INF or NaN */
	if(ix>=0x7ff00000) {
	    if (jx>=0) return one/x+one;    /* tanh(+-inf)=+-1 */
	    else       return one/x-one;    /* tanh(NaN) = NaN */
	}

    /* |x| < 22 */
	if (ix < 0x40360000) {		/* |x|<22 */
	    if (ix<0x3e300000) {	/* |x|<2**-28 */
		if(huge+x>one) return x; /* tanh(tiny) = tiny with inexact */
	    }
	    if (ix>=0x3ff00000) {	/* |x|>=1  */
		t = expm1(two*fabs(x));
		z = one - two/(t+two);
	    } else {
	        t = expm1(-two*fabs(x));
	        z= -t/(t+two);
	    }
    /* |x| >= 22, return +-1 */
	} else {
	    z = one - tiny;		/* raise inexact flag */
	}
	return (jx>=0)? z: -z;
}

#if (LDBL_MANT_DIG == 53)
__weak_reference(tanh, tanhl);
#endif
