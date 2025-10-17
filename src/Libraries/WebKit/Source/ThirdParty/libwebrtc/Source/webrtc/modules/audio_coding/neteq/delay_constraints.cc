/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 16, 2023.
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
#include "modules/audio_coding/neteq/delay_constraints.h"

#include <algorithm>

#include "rtc_base/logging.h"
#include "rtc_base/numerics/safe_minmax.h"

namespace webrtc {

constexpr int kMinBaseMinimumDelayMs = 0;
constexpr int kMaxBaseMinimumDelayMs = 10000;

DelayConstraints::DelayConstraints(int max_packets_in_buffer,
                                   int base_minimum_delay_ms)
    : max_packets_in_buffer_(max_packets_in_buffer),
      base_minimum_delay_ms_(base_minimum_delay_ms),
      effective_minimum_delay_ms_(base_minimum_delay_ms),
      minimum_delay_ms_(0),
      maximum_delay_ms_(0) {}

int DelayConstraints::Clamp(int delay_ms) const {
  delay_ms = std::max(delay_ms, effective_minimum_delay_ms_);
  if (maximum_delay_ms_ > 0) {
    delay_ms = std::min(delay_ms, maximum_delay_ms_);
  }
  if (packet_len_ms_ > 0) {
    // Limit to 75% of maximum buffer size.
    delay_ms =
        std::min(delay_ms, 3 * max_packets_in_buffer_ * packet_len_ms_ / 4);
  }
  return delay_ms;
}

bool DelayConstraints::SetPacketAudioLength(int length_ms) {
  if (length_ms <= 0) {
    RTC_LOG_F(LS_ERROR) << "length_ms = " << length_ms;
    return false;
  }
  packet_len_ms_ = length_ms;
  return true;
}

bool DelayConstraints::IsValidMinimumDelay(int delay_ms) const {
  return 0 <= delay_ms && delay_ms <= MinimumDelayUpperBound();
}

bool DelayConstraints::IsValidBaseMinimumDelay(int delay_ms) const {
  return kMinBaseMinimumDelayMs <= delay_ms &&
         delay_ms <= kMaxBaseMinimumDelayMs;
}

bool DelayConstraints::SetMinimumDelay(int delay_ms) {
  if (!IsValidMinimumDelay(delay_ms)) {
    return false;
  }

  minimum_delay_ms_ = delay_ms;
  UpdateEffectiveMinimumDelay();
  return true;
}

bool DelayConstraints::SetMaximumDelay(int delay_ms) {
  // If `delay_ms` is zero then it unsets the maximum delay and target level is
  // unconstrained by maximum delay.
  if (delay_ms != 0 && delay_ms < minimum_delay_ms_) {
    // Maximum delay shouldn't be less than minimum delay or less than a packet.
    return false;
  }

  maximum_delay_ms_ = delay_ms;
  UpdateEffectiveMinimumDelay();
  return true;
}

bool DelayConstraints::SetBaseMinimumDelay(int delay_ms) {
  if (!IsValidBaseMinimumDelay(delay_ms)) {
    return false;
  }

  base_minimum_delay_ms_ = delay_ms;
  UpdateEffectiveMinimumDelay();
  return true;
}

int DelayConstraints::GetBaseMinimumDelay() const {
  return base_minimum_delay_ms_;
}

void DelayConstraints::UpdateEffectiveMinimumDelay() {
  // Clamp `base_minimum_delay_ms_` into the range which can be effectively
  // used.
  const int base_minimum_delay_ms =
      rtc::SafeClamp(base_minimum_delay_ms_, 0, MinimumDelayUpperBound());
  effective_minimum_delay_ms_ =
      std::max(minimum_delay_ms_, base_minimum_delay_ms);
}

int DelayConstraints::MinimumDelayUpperBound() const {
  // Choose the lowest possible bound discarding 0 cases which mean the value
  // is not set and unconstrained.
  int q75 = max_packets_in_buffer_ * packet_len_ms_ * 3 / 4;
  q75 = q75 > 0 ? q75 : kMaxBaseMinimumDelayMs;
  const int maximum_delay_ms =
      maximum_delay_ms_ > 0 ? maximum_delay_ms_ : kMaxBaseMinimumDelayMs;
  return std::min(maximum_delay_ms, q75);
}

}  // namespace webrtc
