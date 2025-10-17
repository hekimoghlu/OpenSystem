/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 11, 2022.
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
#include <sys/time.h>
#include <errno.h>
#include <time.h>
#include <config.h>

#include "dcethread-private.h"
#include "dcethread-util.h"
#include "dcethread-debug.h"

/*
 *  FUNCTIONAL DESCRIPTION:
 *
 *      Convert a delta timespec to absolute (offset by current time)
 *
 *  FORMAL PARAMETERS:
 *
 *      delta   struct timespec; input delta time
 *
 *      abstime struct timespec; output absolute time
 *
 *  IMPLICIT INPUTS:
 *
 *      current time
 *
 *  IMPLICIT OUTPUTS:
 *
 *      none
 *
 *  FUNCTION VALUE:
 *
 *      0 if successful, else -1 and errno set to error code
 *
 *  SIDE EFFECTS:
 *
 *      none
 */
int
dcethread_get_expiration(struct timespec* delta, struct timespec* abstime)
{
#ifdef HAVE_PTHREAD_GET_EXPIRATION_NP
    return pthread_get_expiration_np(delta, abstime);
#else
    struct timeval now;

    if (delta->tv_nsec >= (1000 * 1000000) || delta->tv_nsec < 0) {
	errno = EINVAL;
	return -1;
    }

    gettimeofday(&now, NULL);

    abstime->tv_nsec    = delta->tv_nsec + (now.tv_usec * 1000);
    abstime->tv_sec     = delta->tv_sec + now.tv_sec;

    if (abstime->tv_nsec >= (1000 * 1000000)) {
	abstime->tv_nsec -= (1000 * 1000000);
	abstime->tv_sec += 1;
    }

    return 0;
#endif /* HAVE_PTHREAD_GET_EXPIRATION_NP */
}
