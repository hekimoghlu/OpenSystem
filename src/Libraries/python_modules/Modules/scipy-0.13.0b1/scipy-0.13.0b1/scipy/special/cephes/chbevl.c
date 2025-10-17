/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 17, 2021.
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
/*							chbevl.c	*/

/*
 * Cephes Math Library Release 2.0:  April, 1987
 * Copyright 1985, 1987 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */

#include <stdio.h>
#include "protos.h"

double chbevl(x, array, n)
double x;
double array[];
int n;
{
    double b0, b1, b2, *p;
    int i;

    p = array;
    b0 = *p++;
    b1 = 0.0;
    i = n - 1;

    do {
	b2 = b1;
	b1 = b0;
	b0 = x * b1 - b2 + *p++;
    }
    while (--i);

    return (0.5 * (b0 - b2));
}
