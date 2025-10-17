/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 8, 2025.
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
/*                                                     stdtri.c
 *
 *     Functional inverse of Student's t distribution
 *
 *
 *
 * SYNOPSIS:
 *
 * double p, t, stdtri();
 * int k;
 *
 * t = stdtri( k, p );
 *
 *
 * DESCRIPTION:
 *
 * Given probability p, finds the argument t such that stdtr(k,t)
 * is equal to p.
 * 
 * ACCURACY:
 *
 * Tested at random 1 <= k <= 100.  The "domain" refers to p:
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE    .001,.999     25000       5.7e-15     8.0e-16
 *    IEEE    10^-6,.001    25000       2.0e-12     2.9e-14
 */


/*
 * Cephes Math Library Release 2.3:  March, 1995
 * Copyright 1984, 1987, 1995 by Stephen L. Moshier
 */

#include "mconf.h"

extern double MAXNUM, MACHEP;

double stdtr(k, t)
int k;
double t;
{
    double x, rk, z, f, tz, p, xsqk;
    int j;

    if (k <= 0) {
	mtherr("stdtr", DOMAIN);
	return (NPY_NAN);
    }

    if (t == 0)
	return (0.5);

    if (t < -2.0) {
	rk = k;
	z = rk / (rk + t * t);
	p = 0.5 * incbet(0.5 * rk, 0.5, z);
	return (p);
    }

    /*     compute integral from -t to + t */

    if (t < 0)
	x = -t;
    else
	x = t;

    rk = k;			/* degrees of freedom */
    z = 1.0 + (x * x) / rk;

    /* test if k is odd or even */
    if ((k & 1) != 0) {

	/*      computation for odd k   */

	xsqk = x / sqrt(rk);
	p = atan(xsqk);
	if (k > 1) {
	    f = 1.0;
	    tz = 1.0;
	    j = 3;
	    while ((j <= (k - 2)) && ((tz / f) > MACHEP)) {
		tz *= (j - 1) / (z * j);
		f += tz;
		j += 2;
	    }
	    p += f * xsqk / z;
	}
	p *= 2.0 / NPY_PI;
    }


    else {

	/*      computation for even k  */

	f = 1.0;
	tz = 1.0;
	j = 2;

	while ((j <= (k - 2)) && ((tz / f) > MACHEP)) {
	    tz *= (j - 1) / (z * j);
	    f += tz;
	    j += 2;
	}
	p = f * x / sqrt(z * rk);
    }

    /*     common exit     */


    if (t < 0)
	p = -p;			/* note destruction of relative accuracy */

    p = 0.5 + 0.5 * p;
    return (p);
}

double stdtri(k, p)
int k;
double p;
{
    double t, rk, z;
    int rflg;

    if (k <= 0 || p <= 0.0 || p >= 1.0) {
	mtherr("stdtri", DOMAIN);
	return (NPY_NAN);
    }

    rk = k;

    if (p > 0.25 && p < 0.75) {
	if (p == 0.5)
	    return (0.0);
	z = 1.0 - 2.0 * p;
	z = incbi(0.5, 0.5 * rk, fabs(z));
	t = sqrt(rk * z / (1.0 - z));
	if (p < 0.5)
	    t = -t;
	return (t);
    }
    rflg = -1;
    if (p >= 0.5) {
	p = 1.0 - p;
	rflg = 1;
    }
    z = incbi(0.5 * rk, 0.5, 2.0 * p);

    if (MAXNUM * z < rk)
	return (rflg * NPY_INFINITY);
    t = sqrt(rk / z - rk);
    return (rflg * t);
}
