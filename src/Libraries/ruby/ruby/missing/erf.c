/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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
#include <stdio.h>
#include <math.h>

#ifdef _WIN32
# include <float.h>
# if !defined __MINGW32__ || defined __NO_ISOCEXT
#  ifndef isnan
#   define isnan(x) _isnan(x)
#  endif
#  ifndef isinf
#   define isinf(x) (!_finite(x) && !_isnan(x))
#  endif
#  ifndef finite
#   define finite(x) _finite(x)
#  endif
# endif
#endif

static double q_gamma(double, double, double);

/* Incomplete gamma function
   1 / Gamma(a) * Int_0^x exp(-t) t^(a-1) dt  */
static double p_gamma(double a, double x, double loggamma_a)
{
    int k;
    double result, term, previous;

    if (x >= 1 + a) return 1 - q_gamma(a, x, loggamma_a);
    if (x == 0)     return 0;
    result = term = exp(a * log(x) - x - loggamma_a) / a;
    for (k = 1; k < 1000; k++) {
        term *= x / (a + k);
        previous = result;  result += term;
        if (result == previous) return result;
    }
    fprintf(stderr, "erf.c:%d:p_gamma() could not converge.", __LINE__);
    return result;
}

/* Incomplete gamma function
   1 / Gamma(a) * Int_x^inf exp(-t) t^(a-1) dt  */
static double q_gamma(double a, double x, double loggamma_a)
{
    int k;
    double result, w, temp, previous;
    double la = 1, lb = 1 + x - a;  /* Laguerre polynomial */

    if (x < 1 + a) return 1 - p_gamma(a, x, loggamma_a);
    w = exp(a * log(x) - x - loggamma_a);
    result = w / lb;
    for (k = 2; k < 1000; k++) {
        temp = ((k - 1 - a) * (lb - la) + (k + x) * lb) / k;
        la = lb;  lb = temp;
        w *= (k - 1 - a) / k;
        temp = w / (la * lb);
        previous = result;  result += temp;
        if (result == previous) return result;
    }
    fprintf(stderr, "erf.c:%d:q_gamma() could not converge.", __LINE__);
    return result;
}

#define LOG_PI_OVER_2 0.572364942924700087071713675675 /* log_e(PI)/2 */

double erf(double x)
{
    if (!finite(x)) {
        if (isnan(x)) return x;      /* erf(NaN)   = NaN   */
        return (x>0 ? 1.0 : -1.0);   /* erf(+-inf) = +-1.0 */
    }
    if (x >= 0) return   p_gamma(0.5, x * x, LOG_PI_OVER_2);
    else        return - p_gamma(0.5, x * x, LOG_PI_OVER_2);
}

double erfc(double x)
{
    if (!finite(x)) {
        if (isnan(x)) return x;      /* erfc(NaN)   = NaN      */
        return (x>0 ? 0.0 : 2.0);    /* erfc(+-inf) = 0.0, 2.0 */
    }
    if (x >= 0) return  q_gamma(0.5, x * x, LOG_PI_OVER_2);
    else        return  1 + p_gamma(0.5, x * x, LOG_PI_OVER_2);
}
