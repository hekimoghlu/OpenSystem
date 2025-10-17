/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 21, 2023.
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
#include "rtc_base/numerics/event_rate_counter.h"

#include <algorithm>
#include <cmath>

#include "api/units/time_delta.h"
#include "api/units/timestamp.h"

namespace webrtc {

void EventRateCounter::AddEvent(Timestamp event_time) {
  if (first_time_.IsFinite())
    interval_.AddSample(event_time - last_time_);
  first_time_ = std::min(first_time_, event_time);
  last_time_ = std::max(last_time_, event_time);
  event_count_++;
}

void EventRateCounter::AddEvents(EventRateCounter other) {
  first_time_ = std::min(first_time_, other.first_time_);
  last_time_ = std::max(last_time_, other.last_time_);
  event_count_ += other.event_count_;
  interval_.AddSamples(other.interval_);
}

bool EventRateCounter::IsEmpty() const {
  return first_time_ == last_time_;
}

double EventRateCounter::Rate() const {
  if (event_count_ == 0)
    return 0;
  if (event_count_ == 1)
    return NAN;
  return (event_count_ - 1) / (last_time_ - first_time_).seconds<double>();
}

TimeDelta EventRateCounter::TotalDuration() const {
  if (first_time_.IsInfinite()) {
    return TimeDelta::Zero();
  }
  return last_time_ - first_time_;
}
}  // namespace webrtc
