/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 5, 2025.
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
 * Copyright 1984, 1987, 1993 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */

/*     Incomplete elliptic integral of second kind     */

#include "mconf.h"

extern double MACHEP;

double ellie(phi, m)
double phi, m;
{
    double a, b, c, e, temp;
    double lphi, t, E;
    int d, mod, npio2, sign;

    if (m == 0.0)
	return (phi);
    lphi = phi;
    npio2 = floor(lphi / NPY_PI_2);
    if (npio2 & 1)
	npio2 += 1;
    lphi = lphi - npio2 * NPY_PI_2;
    if (lphi < 0.0) {
	lphi = -lphi;
	sign = -1;
    }
    else {
	sign = 1;
    }
    a = 1.0 - m;
    E = ellpe(m);
    if (a == 0.0) {
	temp = sin(lphi);
	goto done;
    }
    t = tan(lphi);
    b = sqrt(a);
    /* Thanks to Brian Fitzgerald <fitzgb@mml0.meche.rpi.edu>
     * for pointing out an instability near odd multiples of pi/2.  */
    if (fabs(t) > 10.0) {
	/* Transform the amplitude */
	e = 1.0 / (b * t);
	/* ... but avoid multiple recursions.  */
	if (fabs(e) < 10.0) {
	    e = atan(e);
	    temp = E + m * sin(lphi) * sin(e) - ellie(e, m);
	    goto done;
	}
    }
    c = sqrt(m);
    a = 1.0;
    d = 1;
    e = 0.0;
    mod = 0;

    while (fabs(c / a) > MACHEP) {
	temp = b / a;
	lphi = lphi + atan(t * temp) + mod * NPY_PI;
	mod = (lphi + NPY_PI_2) / NPY_PI;
	t = t * (1.0 + temp) / (1.0 - temp * t * t);
	c = (a - b) / 2.0;
	temp = sqrt(a * b);
	a = (a + b) / 2.0;
	b = temp;
	d += d;
	e += c * sin(lphi);
    }

    temp = E / ellpk(1.0 - m);
    temp *= (atan(t) + mod * NPY_PI) / (d * a);
    temp += e;

  done:

    if (sign < 0)
	temp = -temp;
    temp += npio2 * E;
    return (temp);
}
