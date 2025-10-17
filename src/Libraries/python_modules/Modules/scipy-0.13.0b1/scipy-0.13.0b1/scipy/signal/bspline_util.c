/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 11, 2024.
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
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

void compute_root_from_lambda(double, double *, double *);


void
compute_root_from_lambda(lambda, r, omega)
     double lambda;
     double *r;
     double *omega;
{
    double xi;
    double tmp, tmp2;

    tmp = sqrt(3 + 144*lambda);
    xi = 1 - 96*lambda + 24*lambda * tmp;
    *omega = atan(sqrt((144*lambda - 1.0)/xi));
    tmp2 = sqrt(xi);
    *r = (24*lambda - 1 - tmp2)/(24*lambda) \
	* sqrt((48*lambda + 24*lambda*tmp))/tmp2;
    return;
}
