/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 16, 2022.
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
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include <config.h>

#ifndef HAVE_NANOSLEEP

#include <sys/types.h>
#include <sys/time.h>
#ifdef HAVE_SYS_SELECT_H
#include <sys/select.h>
#endif /* HAVE_SYS_SELECT_H */
#include <time.h>
#include <errno.h>

#include "sudo_compat.h"
#include "sudo_util.h"

int
sudo_nanosleep(const struct timespec *ts, struct timespec *rts)
{
    struct timeval timeout, endtime, now;
    int rval;

    if (ts->tv_sec == 0 && ts->tv_nsec < 1000) {
	timeout.tv_sec = 0;
	timeout.tv_usec = 1;
    } else {
	TIMESPEC_TO_TIMEVAL(&timeout, ts);
    }
    if (rts != NULL) {
	if (gettimeofday(&endtime, NULL) == -1)
	    return -1;
	sudo_timevaladd(&endtime, &timeout, &endtime);
    }
    rval = select(0, NULL, NULL, NULL, &timeout);
    if (rts != NULL && rval == -1 && errno == EINTR) {
	if (gettimeofday(&now, NULL) == -1)
	    return -1;
	sudo_timevalsub(&endtime, &now, &endtime);
	TIMEVAL_TO_TIMESPEC(&endtime, rts);
    }
    return rval;
}
#endif /* HAVE_NANOSLEEP */
