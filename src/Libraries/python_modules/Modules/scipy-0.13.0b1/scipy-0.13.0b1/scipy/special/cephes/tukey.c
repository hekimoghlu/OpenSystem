/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 30, 2024.
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
#include <math.h>

#define SMALLVAL 1e-4
#define EPS 1.0e-14
#define MAXCOUNT 60

double tukeylambdacdf(double x, double lmbda)
{
    double pmin, pmid, pmax, plow, phigh, xeval;
    int count;

    xeval = 1.0 / lmbda;
    if (lmbda > 0.0) {
	if (x < (-xeval))
	    return 0.0;
	if (x > xeval)
	    return 1.0;
    }

    if ((-SMALLVAL < lmbda) && (lmbda < SMALLVAL)) {
	if (x >= 0)
	    return 1.0 / (1.0 + exp(-x));
	else
	    return exp(x) / (1.0 + exp(x));
    }

    pmin = 0.0;
    pmid = 0.5;
    pmax = 1.0;
    plow = pmin;
    phigh = pmax;
    count = 0;

    while ((count < MAXCOUNT) && (fabs(pmid - plow) > EPS)) {
	xeval = (pow(pmid, lmbda) - pow(1.0 - pmid, lmbda)) / lmbda;
	if (xeval == x)
	    return pmid;
	if (xeval > x) {
	    phigh = pmid;
	    pmid = (pmid + plow) / 2.0;
	}
	else {
	    plow = pmid;
	    pmid = (pmid + phigh) / 2.0;
	}
	count++;
    }
    return pmid;
}
