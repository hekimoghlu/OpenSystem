/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 8, 2023.
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
#include <sys/limits.h>
#include <fenv.h>
#include <math.h>

#ifndef type
#define type		double
#define	roundit		round
#define dtype		long
#define	DTYPE_MIN	LONG_MIN
#define	DTYPE_MAX	LONG_MAX
#define	fn		lround
#endif

/*
 * If type has more precision than dtype, the endpoints dtype_(min|max) are
 * of the form xxx.5; they are "out of range" because lround() rounds away
 * from 0.  On the other hand, if type has less precision than dtype, then
 * all values that are out of range are integral, so we might as well assume
 * that everything is in range.  At compile time, INRANGE(x) should reduce to
 * two floating-point comparisons in the former case, or TRUE otherwise.
 */
static const type type_min = (type)DTYPE_MIN;
static const type type_max = (type)DTYPE_MAX;
static const type dtype_min = (type)DTYPE_MIN - 0.5;
static const type dtype_max = (type)DTYPE_MAX + 0.5;
#define	INRANGE(x)	(dtype_max - type_max != 0.5 || \
			 ((x) > dtype_min && (x) < dtype_max))

dtype
fn(type x)
{

	if (INRANGE(x)) {
		x = roundit(x);
		return ((dtype)x);
	} else {
		feraiseexcept(FE_INVALID);
		return (DTYPE_MAX);
	}
}
