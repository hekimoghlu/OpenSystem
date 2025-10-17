/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 20, 2022.
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
#include "timediff.h"

#include <limits.h>

/*
 * Converts number of milliseconds into a timeval structure.
 *
 * Return values:
 *    NULL IF tv is NULL or ms < 0 (eg. no timeout -> blocking select)
 *    tv with 0 in both fields IF ms == 0 (eg. 0ms timeout -> polling select)
 *    tv with converted fields IF ms > 0 (eg. >0ms timeout -> waiting select)
 */
struct timeval *curlx_mstotv(struct timeval *tv, timediff_t ms)
{
  if(!tv)
    return NULL;

  if(ms < 0)
    return NULL;

  if(ms > 0) {
    timediff_t tv_sec = ms / 1000;
    timediff_t tv_usec = (ms % 1000) * 1000; /* max=999999 */
#ifdef HAVE_SUSECONDS_T
#if TIMEDIFF_T_MAX > TIME_T_MAX
    /* tv_sec overflow check in case time_t is signed */
    if(tv_sec > TIME_T_MAX)
      tv_sec = TIME_T_MAX;
#endif
    tv->tv_sec = (time_t)tv_sec;
    tv->tv_usec = (suseconds_t)tv_usec;
#elif defined(_WIN32) /* maybe also others in the future */
#if TIMEDIFF_T_MAX > LONG_MAX
    /* tv_sec overflow check on Windows there we know it is long */
    if(tv_sec > LONG_MAX)
      tv_sec = LONG_MAX;
#endif
    tv->tv_sec = (long)tv_sec;
    tv->tv_usec = (long)tv_usec;
#else
#if TIMEDIFF_T_MAX > INT_MAX
    /* tv_sec overflow check in case time_t is signed */
    if(tv_sec > INT_MAX)
      tv_sec = INT_MAX;
#endif
    tv->tv_sec = (int)tv_sec;
    tv->tv_usec = (int)tv_usec;
#endif
  }
  else {
    tv->tv_sec = 0;
    tv->tv_usec = 0;
  }

  return tv;
}

/*
 * Converts a timeval structure into number of milliseconds.
 */
timediff_t curlx_tvtoms(struct timeval *tv)
{
  return (tv->tv_sec*1000) + (timediff_t)(((double)tv->tv_usec)/1000.0);
}
