/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 16, 2023.
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
#pragma once

#include <errno.h>
#include <time.h>
#include <sys/cdefs.h>

#include "private/bionic_constants.h"

bool timespec_from_timeval(timespec& ts, const timeval& tv);
void timespec_from_ms(timespec& ts, const int ms);

void timeval_from_timespec(timeval& tv, const timespec& ts);

void monotonic_time_from_realtime_time(timespec& monotonic_time, const timespec& realtime_time);
void realtime_time_from_monotonic_time(timespec& realtime_time, const timespec& monotonic_time);

static inline int64_t to_ns(const timespec& ts) {
  return ts.tv_sec * NS_PER_S + ts.tv_nsec;
}

static inline int64_t to_us(const timeval& tv) {
  return tv.tv_sec * US_PER_S + tv.tv_usec;
}

static inline int check_timespec(const timespec* ts, bool null_allowed) {
  if (null_allowed && ts == nullptr) {
    return 0;
  }
  // glibc just segfaults if you pass a null timespec.
  // That seems a lot more likely to catch bad code than returning EINVAL.
  if (ts->tv_nsec < 0 || ts->tv_nsec >= NS_PER_S) {
    return EINVAL;
  }
  if (ts->tv_sec < 0) {
    return ETIMEDOUT;
  }
  return 0;
}

#if !defined(__LP64__)
static inline void absolute_timespec_from_timespec(timespec& abs_ts, const timespec& ts, clockid_t clock) {
  clock_gettime(clock, &abs_ts);
  abs_ts.tv_sec += ts.tv_sec;
  abs_ts.tv_nsec += ts.tv_nsec;
  if (abs_ts.tv_nsec >= NS_PER_S) {
    abs_ts.tv_nsec -= NS_PER_S;
    abs_ts.tv_sec++;
  }
}
#endif
