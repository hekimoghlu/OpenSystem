/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 14, 2023.
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
#include "common_video/frame_rate_estimator.h"

#include "rtc_base/time_utils.h"

namespace webrtc {

FrameRateEstimator::FrameRateEstimator(TimeDelta averaging_window)
    : averaging_window_(averaging_window) {}

void FrameRateEstimator::OnFrame(Timestamp time) {
  CullOld(time);
  frame_times_.push_back(time);
}

std::optional<double> FrameRateEstimator::GetAverageFps() const {
  if (frame_times_.size() < 2) {
    return std::nullopt;
  }
  TimeDelta time_span = frame_times_.back() - frame_times_.front();
  if (time_span < TimeDelta::Micros(1)) {
    return std::nullopt;
  }
  TimeDelta avg_frame_interval = time_span / (frame_times_.size() - 1);

  return static_cast<double>(rtc::kNumMicrosecsPerSec) /
         avg_frame_interval.us();
}

std::optional<double> FrameRateEstimator::GetAverageFps(Timestamp now) {
  CullOld(now);
  return GetAverageFps();
}

void FrameRateEstimator::Reset() {
  frame_times_.clear();
}

void FrameRateEstimator::CullOld(Timestamp now) {
  while (!frame_times_.empty() &&
         frame_times_.front() + averaging_window_ < now) {
    frame_times_.pop_front();
  }
}

}  // namespace webrtc
