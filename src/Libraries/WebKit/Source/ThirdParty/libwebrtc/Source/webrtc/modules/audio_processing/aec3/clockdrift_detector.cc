/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 25, 2025.
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
#include "modules/audio_processing/aec3/clockdrift_detector.h"

namespace webrtc {

ClockdriftDetector::ClockdriftDetector()
    : level_(Level::kNone), stability_counter_(0) {
  delay_history_.fill(0);
}

ClockdriftDetector::~ClockdriftDetector() = default;

void ClockdriftDetector::Update(int delay_estimate) {
  if (delay_estimate == delay_history_[0]) {
    // Reset clockdrift level if delay estimate is stable for 7500 blocks (30
    // seconds).
    if (++stability_counter_ > 7500)
      level_ = Level::kNone;
    return;
  }

  stability_counter_ = 0;
  const int d1 = delay_history_[0] - delay_estimate;
  const int d2 = delay_history_[1] - delay_estimate;
  const int d3 = delay_history_[2] - delay_estimate;

  // Patterns recognized as positive clockdrift:
  // [x-3], x-2, x-1, x.
  // [x-3], x-1, x-2, x.
  const bool probable_drift_up =
      (d1 == -1 && d2 == -2) || (d1 == -2 && d2 == -1);
  const bool drift_up = probable_drift_up && d3 == -3;

  // Patterns recognized as negative clockdrift:
  // [x+3], x+2, x+1, x.
  // [x+3], x+1, x+2, x.
  const bool probable_drift_down = (d1 == 1 && d2 == 2) || (d1 == 2 && d2 == 1);
  const bool drift_down = probable_drift_down && d3 == 3;

  // Set clockdrift level.
  if (drift_up || drift_down) {
    level_ = Level::kVerified;
  } else if ((probable_drift_up || probable_drift_down) &&
             level_ == Level::kNone) {
    level_ = Level::kProbable;
  }

  // Shift delay history one step.
  delay_history_[2] = delay_history_[1];
  delay_history_[1] = delay_history_[0];
  delay_history_[0] = delay_estimate;
}
}  // namespace webrtc
