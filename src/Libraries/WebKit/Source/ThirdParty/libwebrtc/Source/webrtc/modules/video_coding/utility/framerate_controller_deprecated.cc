/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 5, 2022.
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
#include "modules/video_coding/utility/framerate_controller_deprecated.h"

#include <stddef.h>

#include <cstdint>

namespace webrtc {

FramerateControllerDeprecated::FramerateControllerDeprecated(
    float target_framerate_fps)
    : min_frame_interval_ms_(0), framerate_estimator_(1000.0, 1000.0) {
  SetTargetRate(target_framerate_fps);
}

void FramerateControllerDeprecated::SetTargetRate(float target_framerate_fps) {
  if (target_framerate_fps_ != target_framerate_fps) {
    framerate_estimator_.Reset();
    if (last_timestamp_ms_) {
      framerate_estimator_.Update(1, *last_timestamp_ms_);
    }

    const size_t target_frame_interval_ms = 1000 / target_framerate_fps;
    target_framerate_fps_ = target_framerate_fps;
    min_frame_interval_ms_ = 85 * target_frame_interval_ms / 100;
  }
}

float FramerateControllerDeprecated::GetTargetRate() {
  return *target_framerate_fps_;
}

void FramerateControllerDeprecated::Reset() {
  framerate_estimator_.Reset();
  last_timestamp_ms_.reset();
}

bool FramerateControllerDeprecated::DropFrame(uint32_t timestamp_ms) const {
  if (timestamp_ms < last_timestamp_ms_) {
    // Timestamp jumps backward. We can't make adequate drop decision. Don't
    // drop this frame. Stats will be reset in AddFrame().
    return false;
  }

  if (Rate(timestamp_ms).value_or(*target_framerate_fps_) >
      target_framerate_fps_) {
    return true;
  }

  if (last_timestamp_ms_) {
    const int64_t diff_ms =
        static_cast<int64_t>(timestamp_ms) - *last_timestamp_ms_;
    if (diff_ms < min_frame_interval_ms_) {
      return true;
    }
  }

  return false;
}

void FramerateControllerDeprecated::AddFrame(uint32_t timestamp_ms) {
  if (timestamp_ms < last_timestamp_ms_) {
    // Timestamp jumps backward.
    Reset();
  }

  framerate_estimator_.Update(1, timestamp_ms);
  last_timestamp_ms_ = timestamp_ms;
}

std::optional<float> FramerateControllerDeprecated::Rate(
    uint32_t timestamp_ms) const {
  return framerate_estimator_.Rate(timestamp_ms);
}

}  // namespace webrtc
