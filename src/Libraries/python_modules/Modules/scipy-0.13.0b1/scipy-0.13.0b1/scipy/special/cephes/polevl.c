/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 11, 2023.
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
 * Cephes Math Library Release 2.1:  December, 1988
 * Copyright 1984, 1987, 1988 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */
#include "protos.h"

double polevl(x, coef, N)
double x;
double coef[];
int N;
{
    double ans;
    int i;
    double *p;

    p = coef;
    ans = *p++;
    i = N;

    do
	ans = ans * x + *p++;
    while (--i);

    return (ans);
}

/*                                                     p1evl() */
/*                                          N
 * Evaluate polynomial when coefficient of x  is 1.0.
 * Otherwise same as polevl.
 */

double p1evl(x, coef, N)
double x;
double coef[];
int N;
{
    double ans;
    double *p;
    int i;

    p = coef;
    ans = x + *p++;
    i = N - 1;

    do
	ans = ans * x + *p++;
    while (--i);

    return (ans);
}
