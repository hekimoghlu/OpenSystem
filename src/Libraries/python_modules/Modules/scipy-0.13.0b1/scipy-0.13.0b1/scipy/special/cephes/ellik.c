/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 15, 2024.
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
 * Cephes Math Library Release 2.0:  April, 1987
 * Copyright 1984, 1987 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */

/*     Incomplete elliptic integral of first kind      */

#include "mconf.h"
extern double MACHEP;

double ellik(phi, m)
double phi, m;
{
    double a, b, c, e, temp, t, K;
    int d, mod, sign, npio2;

    if (m == 0.0)
	return (phi);
    a = 1.0 - m;
    if (a == 0.0) {
	if (fabs(phi) >= NPY_PI_2) {
	    mtherr("ellik", SING);
	    return (NPY_INFINITY);
	}
	return (log(tan((NPY_PI_2 + phi) / 2.0)));
    }
    npio2 = floor(phi / NPY_PI_2);
    if (npio2 & 1)
	npio2 += 1;
    if (npio2) {
	K = ellpk(a);
	phi = phi - npio2 * NPY_PI_2;
    }
    else
	K = 0.0;
    if (phi < 0.0) {
	phi = -phi;
	sign = -1;
    }
    else
	sign = 0;
    b = sqrt(a);
    t = tan(phi);
    if (fabs(t) > 10.0) {
	/* Transform the amplitude */
	e = 1.0 / (b * t);
	/* ... but avoid multiple recursions.  */
	if (fabs(e) < 10.0) {
	    e = atan(e);
	    if (npio2 == 0)
		K = ellpk(a);
	    temp = K - ellik(e, m);
	    goto done;
	}
    }
    a = 1.0;
    c = sqrt(m);
    d = 1;
    mod = 0;

    while (fabs(c / a) > MACHEP) {
	temp = b / a;
	phi = phi + atan(t * temp) + mod * NPY_PI;
	mod = (phi + NPY_PI_2) / NPY_PI;
	t = t * (1.0 + temp) / (1.0 - temp * t * t);
	c = (a - b) / 2.0;
	temp = sqrt(a * b);
	a = (a + b) / 2.0;
	b = temp;
	d += d;
    }

    temp = (atan(t) + mod * NPY_PI) / (d * a);

  done:
    if (sign < 0)
	temp = -temp;
    temp += npio2 * K;
    return (temp);
}
