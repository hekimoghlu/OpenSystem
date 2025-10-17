/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 4, 2025.
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
/*							cpowl
 *
 *	Complex power function
 *
 *
 *
 * SYNOPSIS:
 *
 * long double complex cpowl();
 * long double complex a, z, w;
 *
 * w = cpowl (a, z);
 *
 *
 *
 * DESCRIPTION:
 *
 * Raises complex A to the complex Zth power.
 * Definition is per AMS55 # 4.2.8,
 * analytically equivalent to cpow(a,z) = cexp(z clog(a)).
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE      -10,+10     30000       9.4e-15     1.5e-15
 *
 */

#include <complex.h>
#include <math.h>
#include "math_private.h"

long double complex
cpowl(long double complex a, long double complex z)
{
	long double complex w;
	long double x, y, r, theta, absa, arga;

	x = creall(z);
	y = cimagl(z);
	absa = cabsl(a);
	if (absa == 0.0L) {
		return (CMPLXL(0.0L, 0.0L));
	}
	arga = cargl(a);
	r = powl(absa, x);
	theta = x * arga;
	if (y != 0.0L) {
		r = r * expl(-y * arga);
		theta = theta + y * logl(absa);
	}
	w = CMPLXL(r * cosl(theta), r * sinl(theta));
	return (w);
}
