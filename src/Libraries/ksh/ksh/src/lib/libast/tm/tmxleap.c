/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 25, 2022.
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
#pragma prototyped
/*
 * Glenn Fowler
 * AT&T Research
 *
 * Time_t conversion support
 */

#include <tmx.h>

/*
 * return t with leap seconds adjusted
 * for direct localtime() access
 */

Time_t
tmxleap(Time_t t)
{
	register Tm_leap_t*	lp;
	uint32_t		sec;

	tmset(tm_info.zone);
	if (tm_info.flags & TM_ADJUST)
	{
		sec = tmxsec(t);
		for (lp = &tm_data.leap[0]; sec < (lp->time - lp->total); lp++);
		t = tmxsns(sec + lp->total, tmxnsec(t));
	}
	return t;
}
