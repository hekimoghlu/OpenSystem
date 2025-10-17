/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 17, 2025.
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
 * Copyright 1984, 1987, 2000 by Stephen L. Moshier
 */

#include "mconf.h"
extern double MAXLOG;

double yn(n, x)
int n;
double x;
{
    double an, anm1, anm2, r;
    int k, sign;

    if (n < 0) {
	n = -n;
	if ((n & 1) == 0)	/* -1**n */
	    sign = 1;
	else
	    sign = -1;
    }
    else
	sign = 1;


    if (n == 0)
	return (sign * y0(x));
    if (n == 1)
	return (sign * y1(x));

    /* test for overflow */
    if (x == 0.0) {
	mtherr("yn", SING);
	return -NPY_INFINITY * sign;
    }
    else if (x < 0.0) {
	mtherr("yn", DOMAIN);
	return NPY_NAN;
    }

    /* forward recurrence on n */

    anm2 = y0(x);
    anm1 = y1(x);
    k = 1;
    r = 2 * k;
    do {
	an = r * anm1 / x - anm2;
	anm2 = anm1;
	anm1 = an;
	r += 2.0;
	++k;
    }
    while (k < n);


    return (sign * an);
}
