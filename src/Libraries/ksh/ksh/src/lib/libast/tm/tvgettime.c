/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 5, 2024.
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

#include "FEATURE/tvlib"

int
tvgettime(Tv_t* tv)
{

#if _lib_clock_gettime && defined(CLOCK_REALTIME)

	struct timespec			s;

	clock_gettime(CLOCK_REALTIME, &s);
	tv->tv_sec = s.tv_sec;
	tv->tv_nsec = s.tv_nsec;

#else

#if defined(tmgettimeofday)

	struct timeval			v;

	tmgettimeofday(&v);
	tv->tv_sec = v.tv_sec;
	tv->tv_nsec = v.tv_usec * 1000;

#else

	static time_t			s;
	static uint32_t			n;

	if ((tv->tv_sec = time(NiL)) != s)
	{
		s = tv->tv_sec;
		n = 0;
	}
	else
		n += 1000;
	tv->tv_nsec = n;

#endif

#endif

	return 0;
}
