/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 23, 2023.
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
#include <config.h>
#include "roken.h"
#ifndef HAVE_GETTIMEOFDAY

#ifdef _WIN32

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
gettimeofday (struct timeval *tp, void *ignore)
{
    FILETIME ft;
    ULARGE_INTEGER li;
    ULONGLONG ull;

    GetSystemTimeAsFileTime(&ft);
    li.LowPart = ft.dwLowDateTime;
    li.HighPart = ft.dwHighDateTime;
    ull = li.QuadPart;

    ull -= 116444736000000000i64;
    ull /= 10i64;               /* ull is now in microseconds */

    tp->tv_usec = (ull % 1000000i64);
    tp->tv_sec  = (ull / 1000000i64);

    return 0;
}

#else

/*
 * Simple gettimeofday that only returns seconds.
 */
ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
gettimeofday (struct timeval *tp, void *ignore)
{
     time_t t;

     t = time(NULL);
     tp->tv_sec  = (long) t;
     tp->tv_usec = 0;
     return 0;
}

#endif  /* !_WIN32 */
#endif
