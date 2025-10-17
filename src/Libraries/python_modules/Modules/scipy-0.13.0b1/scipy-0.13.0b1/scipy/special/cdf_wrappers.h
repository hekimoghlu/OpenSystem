/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 20, 2022.
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
#ifndef _CDF_WRAPPERS_H
#define _CDF_WRAPPERS_H
#ifndef _AMOS_WRAPPERS_H
#include "Python.h"
#endif

#include "sf_error.h"

#include <numpy/npy_math.h>

extern double cdfbet3_wrap(double p, double x, double b);
extern double cdfbet4_wrap(double p, double x, double a);

extern double cdfbin2_wrap(double p, double xn, double pr);
extern double cdfbin3_wrap(double p, double s, double pr);

extern double cdfchi3_wrap(double p, double x);

extern double cdfchn1_wrap(double x, double df, double nc);
extern double cdfchn2_wrap(double p, double df, double nc);
extern double cdfchn3_wrap(double p, double x, double nc);
extern double cdfchn4_wrap(double p, double x, double df);

extern double cdff3_wrap(double p, double f, double dfd);
extern double cdff4_wrap(double p, double f, double dfn);

extern double cdffnc1_wrap(double f, double dfn, double dfd, double nc);
extern double cdffnc2_wrap(double p, double dfn, double dfd, double nc);
extern double cdffnc3_wrap(double p, double f, double dfd, double nc);
extern double cdffnc4_wrap(double p, double f, double dfn, double nc);
extern double cdffnc5_wrap(double p, double f, double dfn, double dfd);

extern double cdfgam1_wrap(double p, double x, double scl);
extern double cdfgam2_wrap(double p, double x, double shp);
extern double cdfgam3_wrap(double p, double x, double scl);
extern double cdfgam4_wrap(double p, double x, double shp);

extern double cdfnbn2_wrap(double p, double xn, double pr);
extern double cdfnbn3_wrap(double p, double s, double pr);

extern double cdfnor3_wrap(double p, double x, double std);
extern double cdfnor4_wrap(double p, double x, double mn);

extern double cdfpoi2_wrap(double p, double xlam);

extern double cdft1_wrap(double p, double t);
extern double cdft2_wrap(double p, double t);
extern double cdft3_wrap(double p, double t);

extern double cdftnc1_wrap(double df, double nc, double t);
extern double cdftnc2_wrap(double df, double nc, double p);
extern double cdftnc3_wrap(double p, double nc, double t);
extern double cdftnc4_wrap(double df, double p, double t);

extern double tukeylambdacdf(double x, double lambda);
#endif
