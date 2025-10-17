/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 20, 2021.
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
#include "f2c.h"
#include <mach/mach_time.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef KR_headers
 integer
cpu_time__(t) real *t;
#else
 integer
cpu_time__(real *t)
#endif
{
	static real ratio;
	static int inited = 0;

	if(!inited) {
	    struct mach_timebase_info info;
	    if(mach_timebase_info(&info) != 0) return 1;
	    ratio = (real)info.numer / ((real)info.denom * NSEC_PER_SEC);
	    inited++;
	}
	*t = ratio * mach_absolute_time();
	return 0;
}
#ifdef __cplusplus
}
#endif
