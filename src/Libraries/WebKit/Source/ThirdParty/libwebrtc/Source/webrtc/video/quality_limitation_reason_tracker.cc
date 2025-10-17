/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 10, 2023.
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
#include "video/quality_limitation_reason_tracker.h"

#include <utility>

#include "rtc_base/checks.h"

namespace webrtc {

QualityLimitationReasonTracker::QualityLimitationReasonTracker(Clock* clock)
    : clock_(clock),
      current_reason_(QualityLimitationReason::kNone),
      current_reason_updated_timestamp_ms_(clock_->TimeInMilliseconds()),
      durations_ms_({std::make_pair(QualityLimitationReason::kNone, 0),
                     std::make_pair(QualityLimitationReason::kCpu, 0),
                     std::make_pair(QualityLimitationReason::kBandwidth, 0),
                     std::make_pair(QualityLimitationReason::kOther, 0)}) {}

QualityLimitationReason QualityLimitationReasonTracker::current_reason() const {
  return current_reason_;
}

void QualityLimitationReasonTracker::SetReason(QualityLimitationReason reason) {
  if (reason == current_reason_)
    return;
  int64_t now_ms = clock_->TimeInMilliseconds();
  durations_ms_[current_reason_] +=
      now_ms - current_reason_updated_timestamp_ms_;
  current_reason_ = reason;
  current_reason_updated_timestamp_ms_ = now_ms;
}

std::map<QualityLimitationReason, int64_t>
QualityLimitationReasonTracker::DurationsMs() const {
  std::map<QualityLimitationReason, int64_t> total_durations_ms = durations_ms_;
  auto it = total_durations_ms.find(current_reason_);
  RTC_DCHECK(it != total_durations_ms.end());
  it->second +=
      clock_->TimeInMilliseconds() - current_reason_updated_timestamp_ms_;
  return total_durations_ms;
}

}  // namespace webrtc
