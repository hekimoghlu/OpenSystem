/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 26, 2025.
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
/* EXP(X)
 * RETURN THE EXPONENTIAL OF X
 * DOUBLE PRECISION (IEEE 53 bits, VAX D FORMAT 56 BITS)
 * CODED IN C BY K.C. NG, 1/19/85;
 * REVISED BY K.C. NG on 2/6/85, 2/15/85, 3/7/85, 3/24/85, 4/16/85, 6/14/86.
 *
 * Required system supported functions:
 *	ldexp(x,n)
 *	copysign(x,y)
 *	isfinite(x)
 *
 * Method:
 *	1. Argument Reduction: given the input x, find r and integer k such
 *	   that
 *	        x = k*ln2 + r,  |r| <= 0.5*ln2.
 *	   r will be represented as r := z+c for better accuracy.
 *
 *	2. Compute exp(r) by
 *
 *		exp(r) = 1 + r + r*R1/(2-R1),
 *	   where
 *		R1 = x - x^2*(p1+x^2*(p2+x^2*(p3+x^2*(p4+p5*x^2)))).
 *
 *	3. exp(x) = 2^k * exp(r) .
 *
 * Special cases:
 *	exp(INF) is INF, exp(NaN) is NaN;
 *	exp(-INF)=  0;
 *	for finite argument, only exp(0)=1 is exact.
 *
 * Accuracy:
 *	exp(x) returns the exponential of x nearly rounded. In a test run
 *	with 1,156,000 random arguments on a VAX, the maximum observed
 *	error was 0.869 ulps (units in the last place).
 */
static const double
    p1 =  1.6666666666666660e-01, /* 0x3fc55555, 0x55555553 */
    p2 = -2.7777777777564776e-03, /* 0xbf66c16c, 0x16c0ac3c */
    p3 =  6.6137564717940088e-05, /* 0x3f11566a, 0xb5c2ba0d */
    p4 = -1.6534060280704225e-06, /* 0xbebbbd53, 0x273e8fb7 */
    p5 =  4.1437773411069054e-08; /* 0x3e663f2a, 0x09c94b6c */

static const double
    ln2hi = 0x1.62e42fee00000p-1,   /* High 32 bits round-down. */
    ln2lo = 0x1.a39ef35793c76p-33;  /* Next 53 bits round-to-nearst. */

static const double
    lnhuge =  0x1.6602b15b7ecf2p9,  /* (DBL_MAX_EXP + 9) * log(2.) */
    lntiny = -0x1.77af8ebeae354p9,  /* (DBL_MIN_EXP - 53 - 10) * log(2.) */
    invln2 =  0x1.71547652b82fep0;  /* 1 / log(2.) */

/* returns exp(r = x + c) for |c| < |x| with no overlap.  */

static double
__exp__D(double x, double c)
{
	double hi, lo, z;
	int k;

	if (x != x)	/* x is NaN. */
		return(x);

	if (x <= lnhuge) {
		if (x >= lntiny) {
			/* argument reduction: x --> x - k*ln2 */
			z = invln2 * x;
			k = z + copysign(0.5, x);

		    	/*
			 * Express (x + c) - k * ln2 as hi - lo.
			 * Let x = hi - lo rounded.
			 */
			hi = x - k * ln2hi;	/* Exact. */
			lo = k * ln2lo - c;
			x = hi - lo;

			/* Return 2^k*[1+x+x*c/(2+c)]  */
			z = x * x;
			c = x - z * (p1 + z * (p2 + z * (p3 + z * (p4 +
			    z * p5))));
			c = (x * c) / (2 - c);

			return (ldexp(1 + (hi - (lo - c)), k));
		} else {
			/* exp(-INF) is 0. exp(-big) underflows to 0.  */
			return (isfinite(x) ? ldexp(1., -5000) : 0);
		}
	} else
	/* exp(INF) is INF, exp(+big#) overflows to INF */
		return (isfinite(x) ? ldexp(1., 5000) : x);
}
