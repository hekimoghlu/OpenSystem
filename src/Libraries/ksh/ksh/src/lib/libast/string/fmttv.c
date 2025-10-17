/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 23, 2024.
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

#include <tv.h>
#include <tm.h>

/*
 * Tv_t fmttime()
 */

char*
fmttv(const char* fmt, Tv_t* tv)
{
	char*	s;
	char*	t;
	int	n;

	s = fmttime(fmt, (time_t)tv->tv_sec);
	if (!tv->tv_nsec || tv->tv_nsec == TV_NSEC_IGNORE)
		return s;
	t = fmtbuf(n = strlen(s) + 11);
	sfsprintf(t, n, "%s.%09lu", s, (unsigned long)tv->tv_nsec);
	return t;
}
