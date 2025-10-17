/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 30, 2022.
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
#include <fenv.h>
#include <math.h>

#ifndef type
#define type		double
#define	roundit		rint
#define dtype		long
#define	fn		lrint
#endif

/*
 * C99 says we should not raise a spurious inexact exception when an
 * invalid exception is raised.  Unfortunately, the set of inputs
 * that overflows depends on the rounding mode when 'dtype' has more
 * significant bits than 'type'.  Hence, we bend over backwards for the
 * sake of correctness; an MD implementation could be more efficient.
 */
dtype
fn(type x)
{
	fenv_t env;
	dtype d;

	feholdexcept(&env);
	d = (dtype)roundit(x);
	if (fetestexcept(FE_INVALID))
		feclearexcept(FE_INEXACT);
	feupdateenv(&env);
	return (d);
}
