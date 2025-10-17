/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 20, 2022.
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
#include "zeros.h"

double
bisect(callback_type f, double xa, double xb, double xtol, double rtol, int iter, default_parameters *params)
{
    int i;
    double dm,xm,fm,fa,fb,tol;

    tol = xtol + rtol*(fabs(xa) + fabs(xb));

    fa = (*f)(xa,params);
    fb = (*f)(xb,params);
    params->funcalls = 2;
    if (fa*fb > 0) {ERROR(params,SIGNERR,0.0);}
    if (fa == 0) return xa;
    if (fb == 0) return xb;
    dm = xb - xa;
    params->iterations = 0;
    for(i=0; i<iter; i++) {
        params->iterations++;
        dm *= .5;
        xm = xa + dm;
        fm = (*f)(xm,params);
        params->funcalls++;
        if (fm*fa >= 0) {
            xa = xm;
        }
        if (fm == 0 || fabs(dm) < tol)
            return xm;
    }
    ERROR(params,CONVERR,xa);
}
