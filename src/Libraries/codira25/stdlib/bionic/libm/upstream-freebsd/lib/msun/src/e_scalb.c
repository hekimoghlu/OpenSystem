/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 4, 2023.
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
 * scalb(x, fn) is provide for
 * passing various standard test suite. One 
 * should use scalbn() instead.
 */

#include "math.h"
#include "math_private.h"

#ifdef _SCALB_INT
double
scalb(double x, int fn)
#else
double
scalb(double x, double fn)
#endif
{
#ifdef _SCALB_INT
	return scalbn(x,fn);
#else
	if (isnan(x)||isnan(fn)) return x*fn;
	if (!finite(fn)) {
	    if(fn>0.0) return x*fn;
	    else       return x/(-fn);
	}
	if (rint(fn)!=fn) return (fn-fn)/(fn-fn);
	if ( fn > 65000.0) return scalbn(x, 65000);
	if (-fn > 65000.0) return scalbn(x,-65000);
	return scalbn(x,(int)fn);
#endif
}
