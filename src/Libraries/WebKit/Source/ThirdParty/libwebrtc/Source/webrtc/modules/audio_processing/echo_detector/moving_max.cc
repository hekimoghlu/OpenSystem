/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 18, 2024.
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
#include "modules/audio_processing/echo_detector/moving_max.h"

#include "rtc_base/checks.h"

namespace webrtc {
namespace {

// Parameter for controlling how fast the estimated maximum decays after the
// previous maximum is no longer valid. With a value of 0.99, the maximum will
// decay to 1% of its former value after 460 updates.
constexpr float kDecayFactor = 0.99f;

}  // namespace

MovingMax::MovingMax(size_t window_size) : window_size_(window_size) {
  RTC_DCHECK_GT(window_size, 0);
}

MovingMax::~MovingMax() {}

void MovingMax::Update(float value) {
  if (counter_ >= window_size_ - 1) {
    max_value_ *= kDecayFactor;
  } else {
    ++counter_;
  }
  if (value > max_value_) {
    max_value_ = value;
    counter_ = 0;
  }
}

float MovingMax::max() const {
  return max_value_;
}

void MovingMax::Clear() {
  max_value_ = 0.f;
  counter_ = 0;
}

}  // namespace webrtc
