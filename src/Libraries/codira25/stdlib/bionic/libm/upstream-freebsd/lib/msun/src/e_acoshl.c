/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 8, 2023.
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
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice 
 * is preserved.
 * ====================================================
 *
 */

/*
 * See e_acosh.c for complete comments.
 *
 * Converted to long double by David Schultz <das@FreeBSD.ORG> and
 * Bruce D. Evans.
 */

#include <float.h>
#ifdef __i386__
#include <ieeefp.h>
#endif

#include "fpmath.h"
#include "math.h"
#include "math_private.h"

/* EXP_LARGE is the threshold above which we use acosh(x) ~= log(2x). */
#if LDBL_MANT_DIG == 64
#define	EXP_LARGE	34
#elif LDBL_MANT_DIG == 113
#define	EXP_LARGE	58
#else
#error "Unsupported long double format"
#endif

#if LDBL_MAX_EXP != 0x4000
/* We also require the usual expsign encoding. */
#error "Unsupported long double format"
#endif

#define	BIAS	(LDBL_MAX_EXP - 1)

static const double
one	= 1.0;

#if LDBL_MANT_DIG == 64
static const union IEEEl2bits
u_ln2 =  LD80C(0xb17217f7d1cf79ac, -1, 6.93147180559945309417e-1L);
#define	ln2	u_ln2.e
#elif LDBL_MANT_DIG == 113
static const long double
ln2 =  6.93147180559945309417232121458176568e-1L;	/* 0x162e42fefa39ef35793c7673007e6.0p-113 */
#else
#error "Unsupported long double format"
#endif

long double
acoshl(long double x)
{
	long double t;
	int16_t hx;

	ENTERI();
	GET_LDBL_EXPSIGN(hx, x);
	if (hx < 0x3fff) {		/* x < 1, or misnormal */
	    RETURNI((x-x)/(x-x));
	} else if (hx >= BIAS + EXP_LARGE) { /* x >= LARGE */
	    if (hx >= 0x7fff) {		/* x is inf, NaN or misnormal */
	        RETURNI(x+x);
	    } else 
		RETURNI(logl(x)+ln2);	/* acosh(huge)=log(2x), or misnormal */
	} else if (hx == 0x3fff && x == 1) {
	    RETURNI(0.0);		/* acosh(1) = 0 */
	} else if (hx >= 0x4000) {	/* LARGE > x >= 2, or misnormal */
	    t=x*x;
	    RETURNI(logl(2.0*x-one/(x+sqrtl(t-one))));
	} else {			/* 1<x<2 */
	    t = x-one;
	    RETURNI(log1pl(t+sqrtl(2.0*t+t*t)));
	}
}

