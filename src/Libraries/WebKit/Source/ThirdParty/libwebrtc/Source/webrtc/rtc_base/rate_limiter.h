/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 7, 2021.
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
#ifndef RTC_BASE_RATE_LIMITER_H_
#define RTC_BASE_RATE_LIMITER_H_

#include <stddef.h>
#include <stdint.h>

#include "rtc_base/rate_statistics.h"
#include "rtc_base/synchronization/mutex.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {

class Clock;

// Class used to limit a bitrate, making sure the average does not exceed a
// maximum as measured over a sliding window. This class is thread safe; all
// methods will acquire (the same) lock befeore executing.
class RateLimiter {
 public:
  RateLimiter(Clock* clock, int64_t max_window_ms);

  RateLimiter() = delete;
  RateLimiter(const RateLimiter&) = delete;
  RateLimiter& operator=(const RateLimiter&) = delete;

  ~RateLimiter();

  // Try to use rate to send bytes. Returns true on success and if so updates
  // current rate.
  bool TryUseRate(size_t packet_size_bytes);

  // Set the maximum bitrate, in bps, that this limiter allows to send.
  void SetMaxRate(uint32_t max_rate_bps);

  // Set the window size over which to measure the current bitrate.
  // For example, irt retransmissions, this is typically the RTT.
  // Returns true on success and false if window_size_ms is out of range.
  bool SetWindowSize(int64_t window_size_ms);

 private:
  Clock* const clock_;
  Mutex lock_;
  RateStatistics current_rate_ RTC_GUARDED_BY(lock_);
  int64_t window_size_ms_ RTC_GUARDED_BY(lock_);
  uint32_t max_rate_bps_ RTC_GUARDED_BY(lock_);
};

}  // namespace webrtc

#endif  // RTC_BASE_RATE_LIMITER_H_
