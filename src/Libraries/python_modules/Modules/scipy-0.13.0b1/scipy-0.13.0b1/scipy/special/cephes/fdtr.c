/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 12, 2024.
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
/*							fdtrc()
 *
 *	Complemented F distribution
 *
 *
 *
 * SYNOPSIS:
 *
 * double df1, df2;
 * double x, y, fdtrc();
 *
 * y = fdtrc( df1, df2, x );
 *
 * DESCRIPTION:
 *
 * Returns the area from x to infinity under the F density
 * function (also known as Snedcor's density or the
 * variance ratio density).
 *
 *
 *                      inf.
 *                       -
 *              1       | |  a-1      b-1
 * 1-P(x)  =  ------    |   t    (1-t)    dt
 *            B(a,b)  | |
 *                     -
 *                      x
 *
 *
 * The incomplete beta integral is used, according to the
 * formula
 *
 *	P(x) = incbet( df2/2, df1/2, (df2/(df2 + df1*x) ).
 *
 *
 * ACCURACY:
 *
 * Tested at random points (a,b,x) in the indicated intervals.
 *                x     a,b                     Relative error:
 * arithmetic  domain  domain     # trials      peak         rms
 *    IEEE      0,1    1,100       100000      3.7e-14     5.9e-16
 *    IEEE      1,5    1,100       100000      8.0e-15     1.6e-15
 *    IEEE      0,1    1,10000     100000      1.8e-11     3.5e-13
 *    IEEE      1,5    1,10000     100000      2.0e-11     3.0e-12
 * See also incbet.c.
 *
 * ERROR MESSAGES:
 *
 *   message         condition      value returned
 * fdtrc domain    a<0, b<0, x<0         0.0
 *
 */
/*							fdtri()
 *
 *	Inverse of F distribution
 *
 *
 *
 * SYNOPSIS:
 *
 * double df1, df2;
 * double x, p, fdtri();
 *
 * x = fdtri( df1, df2, p );
 *
 * DESCRIPTION:
 *
 * Finds the F density argument x such that the integral
 * from -infinity to x of the F density is equal to the
 * given probability p.
 *
 * This is accomplished using the inverse beta integral
 * function and the relations
 *
 *      z = incbi( df2/2, df1/2, p )
 *      x = df2 (1-z) / (df1 z).
 *
 * Note: the following relations hold for the inverse of
 * the uncomplemented F distribution:
 *
 *      z = incbi( df1/2, df2/2, p )
 *      x = df2 z / (df1 (1-z)).
 *
 * ACCURACY:
 *
 * Tested at random points (a,b,p).
 *
 *              a,b                     Relative error:
 * arithmetic  domain     # trials      peak         rms
 *  For p between .001 and 1:
 *    IEEE     1,100       100000      8.3e-15     4.7e-16
 *    IEEE     1,10000     100000      2.1e-11     1.4e-13
 *  For p between 10^-6 and 10^-3:
 *    IEEE     1,100        50000      1.3e-12     8.4e-15
 *    IEEE     1,10000      50000      3.0e-12     4.8e-14
 * See also fdtrc.c.
 *
 * ERROR MESSAGES:
 *
 *   message         condition      value returned
 * fdtri domain   p <= 0 or p > 1       NaN
 *                     v < 1
 *
 */


/*
 * Cephes Math Library Release 2.3:  March, 1995
 * Copyright 1984, 1987, 1995 by Stephen L. Moshier
 */


#include "mconf.h"

double fdtrc(a, b, x)
double a, b;
double x;
{
    double w;

    if ((a < 1.0) || (b < 1.0) || (x < 0.0)) {
	mtherr("fdtrc", DOMAIN);
	return (NPY_NAN);
    }
    w = b / (b + a * x);
    return (incbet(0.5 * b, 0.5 * a, w));
}

double fdtr(a, b, x)
double a, b;
double x;
{
    double w;

    if ((a < 1.0) || (b < 1.0) || (x < 0.0)) {
	mtherr("fdtr", DOMAIN);
	return (NPY_NAN);
    }
    w = a * x;
    w = w / (b + w);
    return (incbet(0.5 * a, 0.5 * b, w));
}


double fdtri(a, b, y)
double a, b;
double y;
{
    double w, x;

    if ((a < 1.0) || (b < 1.0) || (y <= 0.0) || (y > 1.0)) {
	mtherr("fdtri", DOMAIN);
	return (NPY_NAN);
    }
    y = 1.0 - y;
    a = a;
    b = b;
    /* Compute probability for x = 0.5.  */
    w = incbet(0.5 * b, 0.5 * a, 0.5);
    /* If that is greater than y, then the solution w < .5.
     * Otherwise, solve at 1-y to remove cancellation in (b - b*w).  */
    if (w > y || y < 0.001) {
	w = incbi(0.5 * b, 0.5 * a, y);
	x = (b - b * w) / (a * w);
    }
    else {
	w = incbi(0.5 * a, 0.5 * b, 1.0 - y);
	x = b * w / (a * (1.0 - w));
    }
    return (x);
}
