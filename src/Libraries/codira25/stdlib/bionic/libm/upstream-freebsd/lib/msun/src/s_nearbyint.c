/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 31, 2022.
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

/*
 * We save and restore the floating-point environment to avoid raising
 * an inexact exception.  We can get away with using fesetenv()
 * instead of feclearexcept()/feupdateenv() to restore the environment
 * because the only exception defined for rint() is overflow, and
 * rounding can't overflow as long as emax >= p.
 *
 * The volatile keyword is needed below because clang incorrectly assumes
 * that rint won't raise any floating-point exceptions. Declaring ret volatile
 * is sufficient to trick the compiler into doing the right thing.
 */
#define	DECL(type, fn, rint)	\
type				\
fn(type x)			\
{				\
	volatile type ret;	\
	fenv_t env;		\
				\
	fegetenv(&env);		\
	ret = rint(x);		\
	fesetenv(&env);		\
	return (ret);		\
}

DECL(double, nearbyint, rint)
DECL(float, nearbyintf, rintf)
DECL(long double, nearbyintl, rintl)
