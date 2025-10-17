/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 15, 2022.
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
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <sys/stat.h>

#include <errno.h>
#include <fcntl.h>
#include <time.h>
#include <sys/time.h>

#ifndef UTIME_NOW
#define UTIME_NOW -1
#endif

#ifndef UTIME_OMIT
#define UTIME_OMIT -2
#endif
int futimens(int fd, const struct timespec times[2]);

int
futimens(int fd, const struct timespec times[2])
{
	struct timeval now, tv[2], *tvp;
	struct stat sb;

	if (times == NULL || (times[0].tv_nsec == UTIME_NOW &&
	    times[1].tv_nsec == UTIME_NOW))
		tvp = NULL;
	else if (times[0].tv_nsec == UTIME_OMIT &&
	    times[1].tv_nsec == UTIME_OMIT)
		return (0);
	else {
		if ((times[0].tv_nsec < 0 || times[0].tv_nsec > 999999999) &&
		    times[0].tv_nsec != UTIME_NOW &&
		    times[0].tv_nsec != UTIME_OMIT) {
			errno = EINVAL;
			return (-1);
		}
		if ((times[1].tv_nsec < 0 || times[1].tv_nsec > 999999999) &&
		    times[1].tv_nsec != UTIME_NOW &&
		    times[1].tv_nsec != UTIME_OMIT) {
			errno = EINVAL;
			return (-1);
		}
		tv[0].tv_sec = times[0].tv_sec;
		tv[0].tv_usec = times[0].tv_nsec / 1000;
		tv[1].tv_sec = times[1].tv_sec;
		tv[1].tv_usec = times[1].tv_nsec / 1000;
		tvp = tv;
		if (times[0].tv_nsec == UTIME_OMIT ||
		    times[1].tv_nsec == UTIME_OMIT) {
			if (fstat(fd, &sb) == -1)
				return (-1);
			if (times[0].tv_nsec == UTIME_OMIT) {
				tv[0].tv_sec = sb.st_atimespec.tv_sec;
				tv[0].tv_usec = sb.st_atimespec.tv_nsec / 1000;
			}
			if (times[1].tv_nsec == UTIME_OMIT) {
				tv[1].tv_sec = sb.st_mtimespec.tv_sec;
				tv[1].tv_usec = sb.st_mtimespec.tv_nsec / 1000;
			}
		}
		if (times[0].tv_nsec == UTIME_NOW ||
		    times[1].tv_nsec == UTIME_NOW) {
			if (gettimeofday(&now, NULL) == -1)
				return (-1);
			if (times[0].tv_nsec == UTIME_NOW)
				tv[0] = now;
			if (times[1].tv_nsec == UTIME_NOW)
				tv[1] = now;
		}
	}
	return (futimes(fd, tvp));
}
