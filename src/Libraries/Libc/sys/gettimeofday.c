/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 24, 2023.
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
 * Copyright (c) 1995 NeXT Computer, Inc. All Rights Reserved
 *
 * 	File:	libc/sys/gettimeofday.c
 */

#include <time.h>
#include <tzfile.h>
#include <sys/time.h>
#include <errno.h>
#include <unistd.h>

#define	__APPLE_API_PRIVATE
#include <machine/cpu_capabilities.h>
#undef	__APPLE_API_PRIVATE

extern int __gettimeofday(struct timeval *, struct timezone *);
extern int __commpage_gettimeofday(struct timeval *);

int gettimeofday (struct timeval *tp, void *vtzp)
{
	static int validtz = 0;
	static struct timezone cached_tz = {0};
	struct timezone *tzp = (struct timezone *)vtzp;
	struct timeval atv;

	if (tp == NULL) {
		if (tzp == NULL)
			return	(0);
		tp = &atv;
	}

	if (__commpage_gettimeofday(tp)) {		/* first try commpage */
		if (__gettimeofday(tp, NULL) < 0) {	/* if it fails, use syscall */
			return (-1);
		}
	}

	if (tzp) {
	    if (validtz == 0)  {
		struct tm *localtm = localtime ((time_t *)&tp->tv_sec);
		cached_tz.tz_dsttime = localtm->tm_isdst;
		cached_tz.tz_minuteswest =
		    (-localtm->tm_gmtoff / SECSPERMIN) +
		    (localtm->tm_isdst * MINSPERHOUR);
		validtz = 1;
	    }
	    tzp->tz_dsttime = cached_tz.tz_dsttime;
	    tzp->tz_minuteswest = cached_tz.tz_minuteswest;
	}
	return (0);
}
