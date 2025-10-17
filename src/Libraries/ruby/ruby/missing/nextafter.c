/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 15, 2023.
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

#include "ruby/missing.h"

#include <math.h>
#include <float.h>

/* This function doesn't set errno.  It should on POSIX, though. */

double
nextafter(double x, double y)
{
    double x1, x2, d;
    int e;

    if (isnan(x))
        return x;
    if (isnan(y))
        return y;

    if (x == y)
        return y;

    if (x == 0) {
        /* the minimum "subnormal" float */
        x1 = ldexp(0.5, DBL_MIN_EXP - DBL_MANT_DIG + 1);
        if (x1 == 0)
            x1 = DBL_MIN; /* the minimum "normal" float */
        if (0 < y)
            return x1;
        else
            return -x1;
    }

    if (x < 0) {
        if (isinf(x))
            return -DBL_MAX;
        if (x == -DBL_MAX && y < 0 && isinf(y))
            return y;
    }
    else {
        if (isinf(x))
            return DBL_MAX;
        if (x == DBL_MAX && 0 < y && isinf(y))
            return y;
    }

    x1 = frexp(x, &e);

    if (x < y) {
        d = DBL_EPSILON/2;
        if (x1 == -0.5) {
            x1 *= 2;
            e--;
        }
    }
    else {
        d = -DBL_EPSILON/2;
        if (x1 == 0.5) {
            x1 *= 2;
            e--;
        }
    }

    if (e < DBL_MIN_EXP) {
        d = ldexp(d, DBL_MIN_EXP-e);
    }

    x2 = x1 + d;

    if (x2 == 0.0) {
        if (x1 < 0)
            return -0.0;
        else
            return +0.0;
    }

    return ldexp(x2, e);
}
