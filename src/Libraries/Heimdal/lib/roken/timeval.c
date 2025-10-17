/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 14, 2024.
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
 * Timeval stuff
 */

#include <config.h>

#include "roken.h"

/*
 * Make `t1' consistent.
 */

ROKEN_LIB_FUNCTION void ROKEN_LIB_CALL
timevalfix(struct timeval *t1)
{
    if (t1->tv_usec < 0) {
        t1->tv_sec--;
        t1->tv_usec += 1000000;
    }
    if (t1->tv_usec >= 1000000) {
        t1->tv_sec++;
        t1->tv_usec -= 1000000;
    }
}

/*
 * t1 += t2
 */

ROKEN_LIB_FUNCTION void ROKEN_LIB_CALL
timevaladd(struct timeval *t1, const struct timeval *t2)
{
    t1->tv_sec  += t2->tv_sec;
    t1->tv_usec += t2->tv_usec;
    timevalfix(t1);
}

/*
 * t1 -= t2
 */

ROKEN_LIB_FUNCTION void ROKEN_LIB_CALL
timevalsub(struct timeval *t1, const struct timeval *t2)
{
    t1->tv_sec  -= t2->tv_sec;
    t1->tv_usec -= t2->tv_usec;
    timevalfix(t1);
}
