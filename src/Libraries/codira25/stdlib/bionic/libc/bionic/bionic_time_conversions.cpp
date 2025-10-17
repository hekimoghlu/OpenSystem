/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 23, 2022.
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
#include "private/bionic_time_conversions.h"

#include "private/bionic_constants.h"

bool timespec_from_timeval(timespec& ts, const timeval& tv) {
  // Whole seconds can just be copied.
  ts.tv_sec = tv.tv_sec;

  // But we might overflow when converting microseconds to nanoseconds.
  if (tv.tv_usec >= 1000000 || tv.tv_usec < 0) {
    return false;
  }
  ts.tv_nsec = tv.tv_usec * 1000;
  return true;
}

void timespec_from_ms(timespec& ts, const int ms) {
  ts.tv_sec = ms / 1000;
  ts.tv_nsec = (ms % 1000) * 1000000;
}

void timeval_from_timespec(timeval& tv, const timespec& ts) {
  tv.tv_sec = ts.tv_sec;
  tv.tv_usec = ts.tv_nsec / 1000;
}

static void convert_timespec_clocks(timespec& new_time, clockid_t new_clockbase,
                                    const timespec& old_time, clockid_t old_clockbase) {
  // get reference clocks
  timespec new_clock;
  clock_gettime(new_clockbase, &new_clock);
  timespec old_clock;
  clock_gettime(old_clockbase, &old_clock);

  // compute new time by moving old delta to the new clock.
  new_time.tv_sec = old_time.tv_sec - old_clock.tv_sec + new_clock.tv_sec;
  new_time.tv_nsec = old_time.tv_nsec - old_clock.tv_nsec + new_clock.tv_nsec;

  // correct nsec to second wrap.
  if (new_time.tv_nsec >= NS_PER_S) {
    new_time.tv_nsec -= NS_PER_S;
    new_time.tv_sec += 1;
  } else if (new_time.tv_nsec < 0) {
    new_time.tv_nsec += NS_PER_S;
    new_time.tv_sec -= 1;
  }
}

void monotonic_time_from_realtime_time(timespec& monotonic_time, const timespec& realtime_time) {
  convert_timespec_clocks(monotonic_time, CLOCK_MONOTONIC, realtime_time, CLOCK_REALTIME);
}

void realtime_time_from_monotonic_time(timespec& realtime_time, const timespec& monotonic_time) {
  convert_timespec_clocks(realtime_time, CLOCK_REALTIME, monotonic_time, CLOCK_MONOTONIC);
}
