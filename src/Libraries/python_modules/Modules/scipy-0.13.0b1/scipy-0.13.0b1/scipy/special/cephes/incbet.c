/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 11, 2021.
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
 * Cephes Math Library, Release 2.3:  March, 1995
 * Copyright 1984, 1995 by Stephen L. Moshier
 */

#include "mconf.h"

#ifdef DEC
#define MAXGAM 34.84425627277176174
#else
#define MAXGAM 171.624376956302725
#endif

extern double MACHEP, MINLOG, MAXLOG;

static double big = 4.503599627370496e15;
static double biginv = 2.22044604925031308085e-16;

static double incbcf(double a, double b, double x);
static double incbd(double a, double b, double x);
static double pseries(double a, double b, double x);

double incbet(aa, bb, xx)
double aa, bb, xx;
{
    double a, b, t, x, xc, w, y;
    int flag;

    if (aa <= 0.0 || bb <= 0.0)
	goto domerr;

    if ((xx <= 0.0) || (xx >= 1.0)) {
	if (xx == 0.0)
	    return (0.0);
	if (xx == 1.0)
	    return (1.0);
      domerr:
	mtherr("incbet", DOMAIN);
	return (NPY_NAN);
    }

    flag = 0;
    if ((bb * xx) <= 1.0 && xx <= 0.95) {
	t = pseries(aa, bb, xx);
	goto done;
    }

    w = 1.0 - xx;

    /* Reverse a and b if x is greater than the mean. */
    if (xx > (aa / (aa + bb))) {
	flag = 1;
	a = bb;
	b = aa;
	xc = xx;
	x = w;
    }
    else {
	a = aa;
	b = bb;
	xc = w;
	x = xx;
    }

    if (flag == 1 && (b * x) <= 1.0 && x <= 0.95) {
	t = pseries(a, b, x);
	goto done;
    }

    /* Choose expansion for better convergence. */
    y = x * (a + b - 2.0) - (a - 1.0);
    if (y < 0.0)
	w = incbcf(a, b, x);
    else
	w = incbd(a, b, x) / xc;

    /* Multiply w by the factor
     * a      b   _             _     _
     * x  (1-x)   | (a+b) / ( a | (a) | (b) ) .   */

    y = a * log(x);
    t = b * log(xc);
    if ((a + b) < MAXGAM && fabs(y) < MAXLOG && fabs(t) < MAXLOG) {
	t = pow(xc, b);
	t *= pow(x, a);
	t /= a;
	t *= w;
	t *= Gamma(a + b) / (Gamma(a) * Gamma(b));
	goto done;
    }
    /* Resort to logarithms.  */
    y += t + lgam(a + b) - lgam(a) - lgam(b);
    y += log(w / a);
    if (y < MINLOG)
	t = 0.0;
    else
	t = exp(y);

  done:

    if (flag == 1) {
	if (t <= MACHEP)
	    t = 1.0 - MACHEP;
	else
	    t = 1.0 - t;
    }
    return (t);
}

/* Continued fraction expansion #1
 * for incomplete beta integral
 */

static double incbcf(a, b, x)
double a, b, x;
{
    double xk, pk, pkm1, pkm2, qk, qkm1, qkm2;
    double k1, k2, k3, k4, k5, k6, k7, k8;
    double r, t, ans, thresh;
    int n;

    k1 = a;
    k2 = a + b;
    k3 = a;
    k4 = a + 1.0;
    k5 = 1.0;
    k6 = b - 1.0;
    k7 = k4;
    k8 = a + 2.0;

    pkm2 = 0.0;
    qkm2 = 1.0;
    pkm1 = 1.0;
    qkm1 = 1.0;
    ans = 1.0;
    r = 1.0;
    n = 0;
    thresh = 3.0 * MACHEP;
    do {

	xk = -(x * k1 * k2) / (k3 * k4);
	pk = pkm1 + pkm2 * xk;
	qk = qkm1 + qkm2 * xk;
	pkm2 = pkm1;
	pkm1 = pk;
	qkm2 = qkm1;
	qkm1 = qk;

	xk = (x * k5 * k6) / (k7 * k8);
	pk = pkm1 + pkm2 * xk;
	qk = qkm1 + qkm2 * xk;
	pkm2 = pkm1;
	pkm1 = pk;
	qkm2 = qkm1;
	qkm1 = qk;

	if (qk != 0)
	    r = pk / qk;
	if (r != 0) {
	    t = fabs((ans - r) / r);
	    ans = r;
	}
	else
	    t = 1.0;

	if (t < thresh)
	    goto cdone;

	k1 += 1.0;
	k2 += 1.0;
	k3 += 2.0;
	k4 += 2.0;
	k5 += 1.0;
	k6 -= 1.0;
	k7 += 2.0;
	k8 += 2.0;

	if ((fabs(qk) + fabs(pk)) > big) {
	    pkm2 *= biginv;
	    pkm1 *= biginv;
	    qkm2 *= biginv;
	    qkm1 *= biginv;
	}
	if ((fabs(qk) < biginv) || (fabs(pk) < biginv)) {
	    pkm2 *= big;
	    pkm1 *= big;
	    qkm2 *= big;
	    qkm1 *= big;
	}
    }
    while (++n < 300);

  cdone:
    return (ans);
}


/* Continued fraction expansion #2
 * for incomplete beta integral
 */

static double incbd(a, b, x)
double a, b, x;
{
    double xk, pk, pkm1, pkm2, qk, qkm1, qkm2;
    double k1, k2, k3, k4, k5, k6, k7, k8;
    double r, t, ans, z, thresh;
    int n;

    k1 = a;
    k2 = b - 1.0;
    k3 = a;
    k4 = a + 1.0;
    k5 = 1.0;
    k6 = a + b;
    k7 = a + 1.0;;
    k8 = a + 2.0;

    pkm2 = 0.0;
    qkm2 = 1.0;
    pkm1 = 1.0;
    qkm1 = 1.0;
    z = x / (1.0 - x);
    ans = 1.0;
    r = 1.0;
    n = 0;
    thresh = 3.0 * MACHEP;
    do {

	xk = -(z * k1 * k2) / (k3 * k4);
	pk = pkm1 + pkm2 * xk;
	qk = qkm1 + qkm2 * xk;
	pkm2 = pkm1;
	pkm1 = pk;
	qkm2 = qkm1;
	qkm1 = qk;

	xk = (z * k5 * k6) / (k7 * k8);
	pk = pkm1 + pkm2 * xk;
	qk = qkm1 + qkm2 * xk;
	pkm2 = pkm1;
	pkm1 = pk;
	qkm2 = qkm1;
	qkm1 = qk;

	if (qk != 0)
	    r = pk / qk;
	if (r != 0) {
	    t = fabs((ans - r) / r);
	    ans = r;
	}
	else
	    t = 1.0;

	if (t < thresh)
	    goto cdone;

	k1 += 1.0;
	k2 -= 1.0;
	k3 += 2.0;
	k4 += 2.0;
	k5 += 1.0;
	k6 += 1.0;
	k7 += 2.0;
	k8 += 2.0;

	if ((fabs(qk) + fabs(pk)) > big) {
	    pkm2 *= biginv;
	    pkm1 *= biginv;
	    qkm2 *= biginv;
	    qkm1 *= biginv;
	}
	if ((fabs(qk) < biginv) || (fabs(pk) < biginv)) {
	    pkm2 *= big;
	    pkm1 *= big;
	    qkm2 *= big;
	    qkm1 *= big;
	}
    }
    while (++n < 300);
  cdone:
    return (ans);
}

/* Power series for incomplete beta integral.
 * Use when b*x is small and x not too close to 1.  */

static double pseries(a, b, x)
double a, b, x;
{
    double s, t, u, v, n, t1, z, ai;

    ai = 1.0 / a;
    u = (1.0 - b) * x;
    v = u / (a + 1.0);
    t1 = v;
    t = u;
    n = 2.0;
    s = 0.0;
    z = MACHEP * ai;
    while (fabs(v) > z) {
	u = (n - b) * x / n;
	t *= u;
	v = t / (a + n);
	s += v;
	n += 1.0;
    }
    s += t1;
    s += ai;

    u = a * log(x);
    if ((a + b) < MAXGAM && fabs(u) < MAXLOG) {
	t = Gamma(a + b) / (Gamma(a) * Gamma(b));
	s = s * t * pow(x, a);
    }
    else {
	t = lgam(a + b) - lgam(a) - lgam(b) + u + log(s);
	if (t < MINLOG)
	    s = 0.0;
	else
	    s = exp(t);
    }
    return (s);
}
