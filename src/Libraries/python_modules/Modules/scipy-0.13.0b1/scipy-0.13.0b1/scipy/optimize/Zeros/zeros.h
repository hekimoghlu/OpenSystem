/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 3, 2023.
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
/* Modified to not depend on Python everywhere by Travis Oliphant.
 */


#ifndef ZEROS_H
#define ZEROS_H

#define ZEROS_PARAM_HEAD int funcalls; int iterations; int error_num

typedef struct {
    ZEROS_PARAM_HEAD;
} default_parameters;

static double dminarg1,dminarg2;
#define DMIN(a,b) (dminarg1=(a),dminarg2=(b),(dminarg1) < (dminarg2) ?\
        (dminarg1) : (dminarg2))

#define SIGN(a)   ((a) > 0.0 ? 1.0 : -1.0)
#define ERROR(params,num,val) (params)->error_num=(num); return (val)
#define SIGNERR -1
#define CONVERR -2

typedef double (*callback_type)(double,void*);
typedef double (*solver_type)(callback_type, double, double, double, double, int,default_parameters*);

extern double bisect(callback_type f, double xa, double xb, double xtol, double rtol, int iter, default_parameters *params);
extern double ridder(callback_type f, double xa, double xb, double xtol, double rtol, int iter, default_parameters *params);
extern double brenth(callback_type f, double xa, double xb, double xtol, double rtol, int iter, default_parameters *params);
extern double brentq(callback_type f, double xa, double xb, double xtol, double rtol, int iter, default_parameters *params);


extern double fabs(double);
extern double sqrt(double);

#endif
