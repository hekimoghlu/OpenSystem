/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 15, 2024.
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
 * double logb(x)
 * IEEE 754 logb. Included to pass IEEE test suite. Not recommend.
 * Use ilogb instead.
 */

#include <float.h>

#include "math.h"
#include "math_private.h"

static const double
two54 = 1.80143985094819840000e+16;	/* 43500000 00000000 */

double
logb(double x)
{
	int32_t lx,ix;
	EXTRACT_WORDS(ix,lx,x);
	ix &= 0x7fffffff;			/* high |x| */
	if((ix|lx)==0) return -1.0/fabs(x);
	if(ix>=0x7ff00000) return x*x;
	if(ix<0x00100000) {
		x *= two54;		 /* convert subnormal x to normal */
		GET_HIGH_WORD(ix,x);
		ix &= 0x7fffffff;
		return (double) ((ix>>20)-1023-54);
	} else
		return (double) ((ix>>20)-1023);
}

#if (LDBL_MANT_DIG == 53)
__weak_reference(logb, logbl);
#endif
