/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 19, 2021.
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
#include "db_config.h"

#include "db_int.h"

/*
 * __os_gettime --
 *	Return the current time-of-day clock in seconds and nanoseconds.
 *
 * PUBLIC: void __os_gettime __P((ENV *, db_timespec *, int));
 */
void
__os_gettime(env, tp, monotonic)
	ENV *env;
	db_timespec *tp;
	int monotonic;
{
	const char *sc;
	int ret;

#if defined(HAVE_CLOCK_GETTIME)
#if defined(HAVE_CLOCK_MONOTONIC)
	if (monotonic)
		RETRY_CHK((clock_gettime(
		    CLOCK_MONOTONIC, (struct timespec *)tp)), ret);
	else
#endif
		RETRY_CHK((clock_gettime(
		    CLOCK_REALTIME, (struct timespec *)tp)), ret);

	RETRY_CHK((clock_gettime(CLOCK_REALTIME, (struct timespec *)tp)), ret);
	if (ret != 0) {
		sc = "clock_gettime";
		goto err;
	}
#elif defined(HAVE_GETTIMEOFDAY)
	struct timeval v;

	RETRY_CHK((gettimeofday(&v, NULL)), ret);
	if (ret != 0) {
		sc = "gettimeofday";
		goto err;
	}

	tp->tv_sec = v.tv_sec;
	tp->tv_nsec = v.tv_usec * NS_PER_US;
#elif defined(HAVE_TIME)
	time_t now;

	RETRY_CHK((time(&now) == (time_t)-1 ? 1 : 0), ret);
	if (ret != 0) {
		sc = "time";
		goto err;
	}

	tp->tv_sec = now;
	tp->tv_nsec = 0;
#else
	NO AVAILABLE CLOCK IMPLEMENTATION
#endif
	COMPQUIET(monotonic, 0);
	return;

err:	__db_syserr(env, ret, "%s", sc);
	(void)__env_panic(env, __os_posix_err(ret));
}
