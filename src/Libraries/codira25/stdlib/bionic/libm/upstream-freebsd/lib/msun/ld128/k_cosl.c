/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 6, 2022.
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
 * ld128 version of k_cos.c.  See ../src/k_cos.c for most comments.
 */

#include "math_private.h"

/*
 * Domain [-0.7854, 0.7854], range ~[-1.17e-39, 1.19e-39]:
 * |cos(x) - c(x))| < 2**-129.3
 *
 * 113-bit precision requires more care than 64-bit precision, since
 * simple methods give a minimax polynomial with coefficient for x^2
 * that is 1 ulp below 0.5, but we want it to be precisely 0.5.  See
 * ../ld80/k_cosl.c for more details.
 */
static const double
one = 1.0;
static const long double
C1 =  4.16666666666666666666666666666666667e-02L,
C2 = -1.38888888888888888888888888888888834e-03L,
C3 =  2.48015873015873015873015873015446795e-05L,
C4 = -2.75573192239858906525573190949988493e-07L,
C5 =  2.08767569878680989792098886701451072e-09L,
C6 = -1.14707455977297247136657111139971865e-11L,
C7 =  4.77947733238738518870113294139830239e-14L,
C8 = -1.56192069685858079920640872925306403e-16L,
C9 =  4.11031762320473354032038893429515732e-19L,
C10= -8.89679121027589608738005163931958096e-22L,
C11=  1.61171797801314301767074036661901531e-24L,
C12= -2.46748624357670948912574279501044295e-27L;

long double
__kernel_cosl(long double x, long double y)
{
	long double hz,z,r,w;

	z  = x*x;
	r  = z*(C1+z*(C2+z*(C3+z*(C4+z*(C5+z*(C6+z*(C7+
	    z*(C8+z*(C9+z*(C10+z*(C11+z*C12)))))))))));
	hz = 0.5*z;
	w  = one-hz;
	return w + (((one-w)-hz) + (z*r-x*y));
}
