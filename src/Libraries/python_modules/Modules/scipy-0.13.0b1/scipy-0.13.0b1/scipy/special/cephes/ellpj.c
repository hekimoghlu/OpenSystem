/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 2, 2025.
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
/*                                                     ellpj.c         */


/*
 * Cephes Math Library Release 2.0:  April, 1987
 * Copyright 1984, 1987 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */

#include "mconf.h"
extern double MACHEP;

int ellpj(u, m, sn, cn, dn, ph)
double u, m;
double *sn, *cn, *dn, *ph;
{
    double ai, b, phi, t, twon;
    double a[9], c[9];
    int i;


    /* Check for special cases */

    if (m < 0.0 || m > 1.0 || npy_isnan(m)) {
	mtherr("ellpj", DOMAIN);
	*sn = NPY_NAN;
	*cn = NPY_NAN;
	*ph = NPY_NAN;
	*dn = NPY_NAN;
	return (-1);
    }
    if (m < 1.0e-9) {
	t = sin(u);
	b = cos(u);
	ai = 0.25 * m * (u - t * b);
	*sn = t - ai * b;
	*cn = b + ai * t;
	*ph = u - ai;
	*dn = 1.0 - 0.5 * m * t * t;
	return (0);
    }

    if (m >= 0.9999999999) {
	ai = 0.25 * (1.0 - m);
	b = cosh(u);
	t = tanh(u);
	phi = 1.0 / b;
	twon = b * sinh(u);
	*sn = t + ai * (twon - u) / (b * b);
	*ph = 2.0 * atan(exp(u)) - NPY_PI_2 + ai * (twon - u) / b;
	ai *= t * phi;
	*cn = phi - ai * (twon - u);
	*dn = phi + ai * (twon + u);
	return (0);
    }


    /*     A. G. M. scale          */
    a[0] = 1.0;
    b = sqrt(1.0 - m);
    c[0] = sqrt(m);
    twon = 1.0;
    i = 0;

    while (fabs(c[i] / a[i]) > MACHEP) {
	if (i > 7) {
	    mtherr("ellpj", OVERFLOW);
	    goto done;
	}
	ai = a[i];
	++i;
	c[i] = (ai - b) / 2.0;
	t = sqrt(ai * b);
	a[i] = (ai + b) / 2.0;
	b = t;
	twon *= 2.0;
    }

  done:

    /* backward recurrence */
    phi = twon * a[i] * u;
    do {
	t = c[i] * sin(phi) / a[i];
	b = phi;
	phi = (asin(t) + phi) / 2.0;
    }
    while (--i);

    *sn = sin(phi);
    t = cos(phi);
    *cn = t;
    *dn = t / cos(phi - b);
    *ph = phi;
    return (0);
}
