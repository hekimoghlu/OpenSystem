/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 2, 2022.
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
#include <errno.h>
#include <float.h>
#include <math.h>
#include "ruby.h"

/* DBL_MANT_DIG must be less than 4 times of bits of int */
#ifndef DBL_MANT_DIG
#define DBL_MANT_DIG 53		/* in this case, at least 12 digit precision */
#endif
#define BIG_CRITERIA_BIT (1<<DBL_MANT_DIG/2)
#if BIG_CRITERIA_BIT > 0
#define BIG_CRITERIA (1.0*BIG_CRITERIA_BIT)
#else
#define BIG_CRITERIA (1.0*(1<<DBL_MANT_DIG/4)*(1<<(DBL_MANT_DIG/2+1-DBL_MANT_DIG/4)))
#endif
#define SMALL_CRITERIA_BIT (1<<(DBL_MANT_DIG/3))
#if SMALL_CRITERIA_BIT > 0
#define SMALL_CRITERIA (1.0/SMALL_CRITERIA_BIT)
#else
#define SMALL_CRITERIA (1.0*(1<<DBL_MANT_DIG/4)*(1<<(DBL_MANT_DIG/3+1-DBL_MANT_DIG/4)))
#endif

#ifndef HAVE_ACOSH
double
acosh(double x)
{
    if (x < 1)
	x = -1;			/* NaN */
    else if (x == 1)
	return 0;
    else if (x > BIG_CRITERIA)
	x += x;
    else
	x += sqrt((x + 1) * (x - 1));
    return log(x);
}
#endif

#ifndef HAVE_ASINH
double
asinh(double x)
{
    int neg = x < 0;
    double z = fabs(x);

    if (z < SMALL_CRITERIA) return x;
    if (z < (1.0/(1<<DBL_MANT_DIG/5))) {
	double x2 = z * z;
	z *= 1 + x2 * (-1.0/6.0 + x2 * 3.0/40.0);
    }
    else if (z > BIG_CRITERIA) {
	z = log(z + z);
    }
    else {
	z = log(z + sqrt(z * z + 1));
    }
    if (neg) z = -z;
    return z;
}
#endif

#ifndef HAVE_ATANH
double
atanh(double x)
{
    int neg = x < 0;
    double z = fabs(x);

    if (z < SMALL_CRITERIA) return x;
    z = log(z > 1 ? -1 : (1 + z) / (1 - z)) / 2;
    if (neg) z = -z;
    if (isinf(z))
#if defined(ERANGE)
	errno = ERANGE;
#elif defined(EDOM)
	errno = EDOM;
#else
	;
#endif
    return z;
}
#endif
