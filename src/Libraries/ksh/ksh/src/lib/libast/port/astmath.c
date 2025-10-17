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
/*
 * used to test if -last requires -lm
 *
 *	arch		-last			-lm
 *	----		-----			---
 *	linux.sparc	sfdlen,sfputd		frexp,ldexp	
 */

#if N >= 8
#define _ISOC99_SOURCE	1
#endif

#include <math.h>

int
main()
{
#if N & 1
	long double	value = 0;
#else
	double		value = 0;
#endif
#if N < 5
	int		exp = 0;
#endif

#if N == 1
	return ldexpl(value, exp) != 0;
#endif
#if N == 2
	return ldexp(value, exp) != 0;
#endif
#if N == 3
	return frexpl(value, &exp) != 0;
#endif
#if N == 4
	return frexp(value, &exp) != 0;
#endif
#if N == 5
	return isnan(value);
#endif
#if N == 6
	return isnan(value);
#endif
#if N == 7
	return copysign(1.0, value) < 0;
#endif
#if N == 8
	return signbit(value);
#endif
}
