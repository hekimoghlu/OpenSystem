/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 3, 2022.
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
#include "rtc_base/rate_limiter.h"

#include <limits>
#include <optional>

#include "system_wrappers/include/clock.h"

namespace webrtc {

RateLimiter::RateLimiter(Clock* clock, int64_t max_window_ms)
    : clock_(clock),
      current_rate_(max_window_ms, RateStatistics::kBpsScale),
      window_size_ms_(max_window_ms),
      max_rate_bps_(std::numeric_limits<uint32_t>::max()) {}

RateLimiter::~RateLimiter() {}

// Usage note: This class is intended be usable in a scenario where different
// threads may call each of the the different method. For instance, a network
// thread trying to send data calling TryUseRate(), the bandwidth estimator
// calling SetMaxRate() and a timed maintenance thread periodically updating
// the RTT.
bool RateLimiter::TryUseRate(size_t packet_size_bytes) {
  MutexLock lock(&lock_);
  int64_t now_ms = clock_->TimeInMilliseconds();
  std::optional<uint32_t> current_rate = current_rate_.Rate(now_ms);
  if (current_rate) {
    // If there is a current rate, check if adding bytes would cause maximum
    // bitrate target to be exceeded. If there is NOT a valid current rate,
    // allow allocating rate even if target is exceeded. This prevents
    // problems
    // at very low rates, where for instance retransmissions would never be
    // allowed due to too high bitrate caused by a single packet.

    size_t bitrate_addition_bps =
        (packet_size_bytes * 8 * 1000) / window_size_ms_;
    if (*current_rate + bitrate_addition_bps > max_rate_bps_)
      return false;
  }

  current_rate_.Update(packet_size_bytes, now_ms);
  return true;
}

void RateLimiter::SetMaxRate(uint32_t max_rate_bps) {
  MutexLock lock(&lock_);
  max_rate_bps_ = max_rate_bps;
}

// Set the window size over which to measure the current bitrate.
// For retransmissions, this is typically the RTT.
bool RateLimiter::SetWindowSize(int64_t window_size_ms) {
  MutexLock lock(&lock_);
  window_size_ms_ = window_size_ms;
  return current_rate_.SetWindowSize(window_size_ms,
                                     clock_->TimeInMilliseconds());
}

}  // namespace webrtc
