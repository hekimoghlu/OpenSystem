/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 18, 2022.
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
#include "rtc_base/frequency_tracker.h"

#include <optional>

#include "api/units/frequency.h"
#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "rtc_base/rate_statistics.h"

namespace webrtc {

FrequencyTracker::FrequencyTracker(TimeDelta max_window_size)
    : impl_(max_window_size.ms(), 1'000'000) {}

std::optional<Frequency> FrequencyTracker::Rate(Timestamp now) const {
  if (std::optional<int64_t> rate = impl_.Rate(now.ms())) {
    return Frequency::MilliHertz(*rate);
  }
  return std::nullopt;
}

void FrequencyTracker::Update(int64_t count, Timestamp now) {
  impl_.Update(count, now.ms());
}

}  // namespace webrtc
