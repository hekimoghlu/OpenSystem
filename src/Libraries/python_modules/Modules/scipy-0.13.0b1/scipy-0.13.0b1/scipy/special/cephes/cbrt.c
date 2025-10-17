/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 15, 2022.
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
/*							cbrt.c  */

/*
 * Cephes Math Library Release 2.2:  January, 1991
 * Copyright 1984, 1991 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */


#include "mconf.h"

static double CBRT2 = 1.2599210498948731647672;
static double CBRT4 = 1.5874010519681994747517;
static double CBRT2I = 0.79370052598409973737585;
static double CBRT4I = 0.62996052494743658238361;

double cbrt(double x)
{
    int e, rem, sign;
    double z;

    if (!npy_isfinite(x))
	return x;
    if (x == 0)
	return (x);
    if (x > 0)
	sign = 1;
    else {
	sign = -1;
	x = -x;
    }

    z = x;
    /* extract power of 2, leaving
     * mantissa between 0.5 and 1
     */
    x = frexp(x, &e);

    /* Approximate cube root of number between .5 and 1,
     * peak relative error = 9.2e-6
     */
    x = (((-1.3466110473359520655053e-1 * x
	   + 5.4664601366395524503440e-1) * x
	  - 9.5438224771509446525043e-1) * x
	 + 1.1399983354717293273738e0) * x + 4.0238979564544752126924e-1;

    /* exponent divided by 3 */
    if (e >= 0) {
	rem = e;
	e /= 3;
	rem -= 3 * e;
	if (rem == 1)
	    x *= CBRT2;
	else if (rem == 2)
	    x *= CBRT4;
    }


    /* argument less than 1 */

    else {
	e = -e;
	rem = e;
	e /= 3;
	rem -= 3 * e;
	if (rem == 1)
	    x *= CBRT2I;
	else if (rem == 2)
	    x *= CBRT4I;
	e = -e;
    }

    /* multiply by power of 2 */
    x = ldexp(x, e);

    /* Newton iteration */
    x -= (x - (z / (x * x))) * 0.33333333333333333333;
#ifdef DEC
    x -= (x - (z / (x * x))) / 3.0;
#else
    x -= (x - (z / (x * x))) * 0.33333333333333333333;
#endif

    if (sign < 0)
	x = -x;
    return (x);
}
