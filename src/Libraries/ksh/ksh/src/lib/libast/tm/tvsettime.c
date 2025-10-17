/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 20, 2025.
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
#include <errno.h>

#include "FEATURE/tvlib"

int
tvsettime(const Tv_t* tv)
{

#if _lib_clock_settime && defined(CLOCK_REALTIME)

	struct timespec			s;

	s.tv_sec = tv->tv_sec;
	s.tv_nsec = tv->tv_nsec;
	return clock_settime(CLOCK_REALTIME, &s);

#else

#if defined(tmsettimeofday)

	struct timeval			v;

	v.tv_sec = tv->tv_sec;
	v.tv_usec = tv->tv_nsec / 1000;
	return tmsettimeofday(&v);

#else

#if _lib_stime

	static time_t			s;

	s = tv->tv_sec + (tv->tv_nsec != 0);
	return stime(s);

#else

	errno = EPERM;
	return -1;

#endif

#endif

#endif

}
