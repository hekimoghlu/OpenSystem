/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 26, 2023.
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

#ifndef C_MISC_MISC_H
#define C_MISC_MISC_H

typedef enum {
  /* An exact solution was found, in which case the first point
     on the interval is the value */
  FSOLVE_EXACT,
  /* Interval width is less than the tolerance */
  FSOLVE_CONVERGED,
  /* Not a bracket */
  FSOLVE_NOT_BRACKET,
  /* Root-finding didn't converge in a set number of iterations. */
  FSOLVE_MAX_ITERATIONS
} fsolve_result_t;

typedef double (*objective_function)(double, void *);

fsolve_result_t false_position(double *a, double *fa, double *b, double *fb,
                       objective_function f, void *f_extra,
                       double abserr, double relerr, double bisect_til,
                       double *best_x, double *best_f, double *errest);

double besselpoly(double a, double lambda, double nu);
double gammaincinv(double a, double x);
double gammasgn(double x);

double struve_h(double v, double x);
double struve_l(double v, double x);
double struve_power_series(double v, double x, int is_h, double *err);
double struve_asymp_large_z(double v, double z, int is_h, double *err);
double struve_bessel_series(double v, double z, int is_h, double *err);

#define gammaincinv_doc """gammaincinv(a, y) returns x such that gammainc(a, x) = y."""

#endif /* C_MISC_MISC_H */
