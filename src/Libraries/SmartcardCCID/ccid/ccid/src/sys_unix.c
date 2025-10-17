/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 7, 2025.
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
/**
 * @file
 * @brief This handles abstract system level calls.
 */

#include "config.h"
#define _GNU_SOURCE /* for secure_getenv(3) */
#include <sys/time.h>
#include <limits.h>
#include <stdlib.h>
#include <time.h>
#ifdef HAVE_GETRANDOM
#include <sys/random.h>
#endif /* HAVE_GETRANDOM */
#include <errno.h>
#include <string.h>
#include <unistd.h>

#include "misc.h"
#include "sys_generic.h"
#include "debuglog.h"

/**
 * @brief Makes the current process sleep for some seconds.
 *
 * @param[in] iTimeVal Number of seconds to sleep.
 */
INTERNAL int SYS_Sleep(int iTimeVal)
{
#ifdef HAVE_NANOSLEEP
	struct timespec mrqtp;
	mrqtp.tv_sec = iTimeVal;
	mrqtp.tv_nsec = 0;

	return nanosleep(&mrqtp, NULL);
#else
	return sleep(iTimeVal);
#endif
}

/**
 * @brief Makes the current process sleep for some microseconds.
 *
 * @param[in] iTimeVal Number of microseconds to sleep.
 */
INTERNAL int SYS_USleep(int iTimeVal)
{
#ifdef HAVE_NANOSLEEP
	struct timespec mrqtp;
	mrqtp.tv_sec = iTimeVal/1000000;
	mrqtp.tv_nsec = (iTimeVal - (mrqtp.tv_sec * 1000000)) * 1000;

	return nanosleep(&mrqtp, NULL);
#else
	struct timeval tv;
	tv.tv_sec  = iTimeVal/1000000;
	tv.tv_usec = iTimeVal - (tv.tv_sec * 1000000);
	return select(0, NULL, NULL, NULL, &tv);
#endif
}

/**
 * Generate a pseudo random number
 *
 * @return a non-negative random number
 *
 * @remark the range is at least up to `2^31`.
 * @remark this is a CSPRNG when `getrandom()` is available, LCG otherwise.
 * @warning SYS_InitRandom() should be called (once) before using this function.
 * @warning not thread safe when system lacks `getrandom()` syscall.
 * @warning not cryptographically secure when system lacks `getrandom()` syscall.
 * @warning if interrupted by a signal, this function may return 0.
 */
INTERNAL int SYS_RandomInt(void)
{
#ifdef HAVE_GETRANDOM
	unsigned int ui = 0;
	unsigned char c[sizeof ui] = {0};
	size_t i;
	ssize_t ret;

	ret = getrandom(c, sizeof c, 0);
	if (-1 == ret)
	{
		Log2(PCSC_LOG_ERROR, "getrandom() failed: %s", strerror(errno));
		return lrand48();
	}
	// this loop avoids trap representations that may occur in the naive solution
	for(i = 0; i < sizeof ui; i++) {
		ui <<= CHAR_BIT;
		ui |= c[i];
	}
	// the casts are for the sake of clarity
	return (int)(ui & (unsigned int)INT_MAX);
#else
	int r = lrand48(); // this is not thread-safe
	return r;
#endif /* HAVE_GETRANDOM */
}

/**
 * Initialize the random generator
 */
INTERNAL void SYS_InitRandom(void)
{
#ifndef HAVE_GETRANDOM
	struct timeval tv;
	struct timezone tz;
	long myseed = 0;

	tz.tz_minuteswest = 0;
	tz.tz_dsttime = 0;
	if (gettimeofday(&tv, &tz) == 0)
	{
		myseed = tv.tv_usec;
	} else
	{
		myseed = (long) time(NULL);
	}

	srand48(myseed);
#endif /* HAVE_GETRANDOM */
}

/**
 * (More) secure version of getenv(3)
 *
 * @param[in] name variable environment name
 *
 * @return value of the environment variable called "name"
 */
INTERNAL const char * SYS_GetEnv(const char *name)
{
#ifdef HAVE_SECURE_GETENV
	return secure_getenv(name);
#else
	/* Otherwise, make sure current process is not tainted by uid or gid
	 * changes */
#ifdef HAVE_issetugid
	if (issetugid())
		return NULL;
#endif
	return getenv(name);
#endif
}

