/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 18, 2023.
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
 * See e_atanh.c for complete comments.
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

/* EXP_TINY is the threshold below which we use atanh(x) ~= x. */
#if LDBL_MANT_DIG == 64
#define	EXP_TINY	-34
#elif LDBL_MANT_DIG == 113
#define	EXP_TINY	-58
#else
#error "Unsupported long double format"
#endif

#if LDBL_MAX_EXP != 0x4000
/* We also require the usual expsign encoding. */
#error "Unsupported long double format"
#endif

#define	BIAS	(LDBL_MAX_EXP - 1)

static const double one = 1.0, huge = 1e300;
static const double zero = 0.0;

long double
atanhl(long double x)
{
	long double t;
	uint16_t hx, ix;

	ENTERI();
	GET_LDBL_EXPSIGN(hx, x);
	ix = hx & 0x7fff;
	if (ix >= 0x3fff)		/* |x| >= 1, or NaN or misnormal */
	    RETURNI(fabsl(x) == 1 ? x / zero : (x - x) / (x - x));
	if (ix < BIAS + EXP_TINY && (huge + x) > zero)
	    RETURNI(x);			/* x is tiny */
	SET_LDBL_EXPSIGN(x, ix);
	if (ix < 0x3ffe) {		/* |x| < 0.5, or misnormal */
	    t = x+x;
	    t = 0.5*log1pl(t+t*x/(one-x));
	} else 
	    t = 0.5*log1pl((x+x)/(one-x));
	RETURNI((hx & 0x8000) == 0 ? t : -t);
}

