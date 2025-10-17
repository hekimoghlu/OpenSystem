/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 19, 2023.
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
/*							cpowf
 *
 *	Complex power function
 *
 *
 *
 * SYNOPSIS:
 *
 * float complex cpowf();
 * float complex a, z, w;
 *
 * w = cpowf (a, z);
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

float complex
cpowf(float complex a, float complex z)
{
	float complex w;
	float x, y, r, theta, absa, arga;

	x = crealf(z);
	y = cimagf(z);
	absa = cabsf (a);
	if (absa == 0.0f) {
		return (CMPLXF(0.0f, 0.0f));
	}
	arga = cargf (a);
	r = powf (absa, x);
	theta = x * arga;
	if (y != 0.0f) {
		r = r * expf (-y * arga);
		theta = theta + y * logf (absa);
	}
	w = CMPLXF(r * cosf (theta), r * sinf (theta));
	return (w);
}
