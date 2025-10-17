/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 1, 2023.
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
 * Cephes Math Library Release 2.3:  March, 1995
 * Copyright 1984, 1987, 1995 by Stephen L. Moshier
 */

#include "mconf.h"
#include <stdio.h>

extern double MACHEP, MAXLOG, MINLOG;

double igami(a, y0)
double a, y0;
{
    double x0, x1, x, yl, yh, y, d, lgm, dithresh;
    int i, dir;

    /* bound the solution */
    x0 = NPY_INFINITY;
    yl = 0;
    x1 = 0;
    yh = 1.0;
    dithresh = 5.0 * MACHEP;

    if ((y0 < 0.0) || (y0 > 1.0) || (a <= 0)) {
	mtherr("igami", DOMAIN);
	return (NPY_NAN);
    }

    if (y0 == 0.0) {
	return (NPY_INFINITY);
    }

    if (y0 == 1.0) {
	return 0.0;
    }

    /* approximation to inverse function */
    d = 1.0 / (9.0 * a);
    y = (1.0 - d - ndtri(y0) * sqrt(d));
    x = a * y * y * y;

    lgm = lgam(a);

    for (i = 0; i < 10; i++) {
	if (x > x0 || x < x1)
	    goto ihalve;
	y = igamc(a, x);
	if (y < yl || y > yh)
	    goto ihalve;
	if (y < y0) {
	    x0 = x;
	    yl = y;
	}
	else {
	    x1 = x;
	    yh = y;
	}
	/* compute the derivative of the function at this point */
	d = (a - 1.0) * log(x) - x - lgm;
	if (d < -MAXLOG)
	    goto ihalve;
	d = -exp(d);
	/* compute the step to the next approximation of x */
	d = (y - y0) / d;
	if (fabs(d / x) < MACHEP)
	    goto done;
	x = x - d;
    }

    /* Resort to interval halving if Newton iteration did not converge. */
  ihalve:

    d = 0.0625;
    if (x0 == NPY_INFINITY) {
	if (x <= 0.0)
	    x = 1.0;
	while (x0 == NPY_INFINITY) {
	    x = (1.0 + d) * x;
	    y = igamc(a, x);
	    if (y < y0) {
		x0 = x;
		yl = y;
		break;
	    }
	    d = d + d;
	}
    }
    d = 0.5;
    dir = 0;

    for (i = 0; i < 400; i++) {
	x = x1 + d * (x0 - x1);
	y = igamc(a, x);
	lgm = (x0 - x1) / (x1 + x0);
	if (fabs(lgm) < dithresh)
	    break;
	lgm = (y - y0) / y0;
	if (fabs(lgm) < dithresh)
	    break;
	if (x <= 0.0)
	    break;
	if (y >= y0) {
	    x1 = x;
	    yh = y;
	    if (dir < 0) {
		dir = 0;
		d = 0.5;
	    }
	    else if (dir > 1)
		d = 0.5 * d + 0.5;
	    else
		d = (y0 - yl) / (yh - yl);
	    dir += 1;
	}
	else {
	    x0 = x;
	    yl = y;
	    if (dir > 0) {
		dir = 0;
		d = 0.5;
	    }
	    else if (dir < -1)
		d = 0.5 * d;
	    else
		d = (y0 - yl) / (yh - yl);
	    dir -= 1;
	}
    }
    if (x == 0.0)
	mtherr("igami", UNDERFLOW);

  done:
    return (x);
}
