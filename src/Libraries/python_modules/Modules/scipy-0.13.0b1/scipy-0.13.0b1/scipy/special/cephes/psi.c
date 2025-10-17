/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 26, 2021.
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
 * Cephes Math Library Release 2.8:  June, 2000
 * Copyright 1984, 1987, 1992, 2000 by Stephen L. Moshier
 */

#include "mconf.h"

#ifdef UNK
static double A[] = {
    8.33333333333333333333E-2,
    -2.10927960927960927961E-2,
    7.57575757575757575758E-3,
    -4.16666666666666666667E-3,
    3.96825396825396825397E-3,
    -8.33333333333333333333E-3,
    8.33333333333333333333E-2
};
#endif

#ifdef DEC
static unsigned short A[] = {
    0037252, 0125252, 0125252, 0125253,
    0136654, 0145314, 0126312, 0146255,
    0036370, 0037017, 0101740, 0174076,
    0136210, 0104210, 0104210, 0104211,
    0036202, 0004040, 0101010, 0020202,
    0136410, 0104210, 0104210, 0104211,
    0037252, 0125252, 0125252, 0125253
};
#endif

#ifdef IBMPC
static unsigned short A[] = {
    0x5555, 0x5555, 0x5555, 0x3fb5,
    0x5996, 0x9599, 0x9959, 0xbf95,
    0x1f08, 0xf07c, 0x07c1, 0x3f7f,
    0x1111, 0x1111, 0x1111, 0xbf71,
    0x0410, 0x1041, 0x4104, 0x3f70,
    0x1111, 0x1111, 0x1111, 0xbf81,
    0x5555, 0x5555, 0x5555, 0x3fb5
};
#endif

#ifdef MIEEE
static unsigned short A[] = {
    0x3fb5, 0x5555, 0x5555, 0x5555,
    0xbf95, 0x9959, 0x9599, 0x5996,
    0x3f7f, 0x07c1, 0xf07c, 0x1f08,
    0xbf71, 0x1111, 0x1111, 0x1111,
    0x3f70, 0x4104, 0x1041, 0x0410,
    0xbf81, 0x1111, 0x1111, 0x1111,
    0x3fb5, 0x5555, 0x5555, 0x5555
};
#endif

double psi(x)
double x;
{
    double p, q, nz, s, w, y, z;
    int i, n, negative;

    negative = 0;
    nz = 0.0;

    if (x <= 0.0) {
	negative = 1;
	q = x;
	p = floor(q);
	if (p == q) {
	    mtherr("psi", SING);
	    return (NPY_INFINITY);
	}
	/* Remove the zeros of tan(NPY_PI x)
	 * by subtracting the nearest integer from x
	 */
	nz = q - p;
	if (nz != 0.5) {
	    if (nz > 0.5) {
		p += 1.0;
		nz = q - p;
	    }
	    nz = NPY_PI / tan(NPY_PI * nz);
	}
	else {
	    nz = 0.0;
	}
	x = 1.0 - x;
    }

    /* check for positive integer up to 10 */
    if ((x <= 10.0) && (x == floor(x))) {
	y = 0.0;
	n = x;
	for (i = 1; i < n; i++) {
	    w = i;
	    y += 1.0 / w;
	}
	y -= NPY_EULER;
	goto done;
    }

    s = x;
    w = 0.0;
    while (s < 10.0) {
	w += 1.0 / s;
	s += 1.0;
    }

    if (s < 1.0e17) {
	z = 1.0 / (s * s);
	y = z * polevl(z, A, 6);
    }
    else
	y = 0.0;

    y = log(s) - (0.5 / s) - y - w;

  done:

    if (negative) {
	y -= nz;
    }

    return (y);
}
