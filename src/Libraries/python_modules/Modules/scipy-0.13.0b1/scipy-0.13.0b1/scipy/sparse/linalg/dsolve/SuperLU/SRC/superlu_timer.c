/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 24, 2024.
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
#ifdef SUN 
/*
 * 	It uses the system call gethrtime(3C), which is accurate to 
 *	nanoseconds. 
*/
#include <sys/time.h>

double SuperLU_timer_() {
    return ( (double)gethrtime() / 1e9 );
}

#elif _WIN32

#include <time.h>

double SuperLU_timer_()
{
    clock_t t;
    t=clock();

    return ((double)t)/CLOCKS_PER_SEC;
}

#else

#ifndef NO_TIMER
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <unistd.h>
#endif

/*! \brief Timer function
 */ 

double SuperLU_timer_()
{
#ifdef NO_TIMER
    /* no sys/times.h on WIN32 */
    double tmp;
    tmp = 0.0;
    /* return (double)(tmp) / CLK_TCK;*/
    return 0.0;
#else
    struct tms use;
    double tmp;
    int clocks_per_sec = sysconf(_SC_CLK_TCK);

    times ( &use );
    tmp = use.tms_utime;
    tmp += use.tms_stime;
    return (double)(tmp) / clocks_per_sec;
#endif
}

#endif

