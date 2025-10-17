/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 11, 2022.
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
/*							gdtrc.c
 *
 *	Complemented Gamma distribution function
 *
 *
 *
 * SYNOPSIS:
 *
 * double a, b, x, y, gdtrc();
 *
 * y = gdtrc( a, b, x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns the integral from x to infinity of the Gamma
 * probability density function:
 *
 *
 *               inf.
 *        b       -
 *       a       | |   b-1  -at
 * y =  -----    |    t    e    dt
 *       -     | |
 *      | (b)   -
 *               x
 *
 *  The incomplete Gamma integral is used, according to the
 * relation
 *
 * y = igamc( b, ax ).
 *
 *
 * ACCURACY:
 *
 * See igamc().
 *
 * ERROR MESSAGES:
 *
 *   message         condition      value returned
 * gdtrc domain         x < 0            0.0
 *
 */

/*                                                     gdtr()  */


/*
 * Cephes Math Library Release 2.3:  March,1995
 * Copyright 1984, 1987, 1995 by Stephen L. Moshier
 */

#include "mconf.h"
double gdtri(double, double, double);

double gdtr(a, b, x)
double a, b, x;
{

    if (x < 0.0) {
	mtherr("gdtr", DOMAIN);
	return (NPY_NAN);
    }
    return (igam(b, a * x));
}


double gdtrc(a, b, x)
double a, b, x;
{

    if (x < 0.0) {
	mtherr("gdtrc", DOMAIN);
	return (NPY_NAN);
    }
    return (igamc(b, a * x));
}


double gdtri(a, b, y)
double a, b, y;
{

    if ((y < 0.0) || (y > 1.0) || (a <= 0.0) || (b < 0.0)) {
	mtherr("gdtri", DOMAIN);
	return (NPY_NAN);
    }

    return (igami(b, 1.0 - y) / a);
}
