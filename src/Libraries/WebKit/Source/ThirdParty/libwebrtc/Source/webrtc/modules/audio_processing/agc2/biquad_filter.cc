/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 23, 2022.
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
#include "modules/audio_processing/agc2/biquad_filter.h"

#include "rtc_base/arraysize.h"

namespace webrtc {

BiQuadFilter::BiQuadFilter(const Config& config)
    : config_(config), state_({}) {}

BiQuadFilter::~BiQuadFilter() = default;

void BiQuadFilter::SetConfig(const Config& config) {
  config_ = config;
  state_ = {};
}

void BiQuadFilter::Reset() {
  state_ = {};
}

void BiQuadFilter::Process(rtc::ArrayView<const float> x,
                           rtc::ArrayView<float> y) {
  RTC_DCHECK_EQ(x.size(), y.size());
  const float config_a0 = config_.a[0];
  const float config_a1 = config_.a[1];
  const float config_b0 = config_.b[0];
  const float config_b1 = config_.b[1];
  const float config_b2 = config_.b[2];
  float state_a0 = state_.a[0];
  float state_a1 = state_.a[1];
  float state_b0 = state_.b[0];
  float state_b1 = state_.b[1];
  for (size_t k = 0, x_size = x.size(); k < x_size; ++k) {
    // Use a temporary variable for `x[k]` to allow in-place processing.
    const float tmp = x[k];
    float y_k = config_b0 * tmp + config_b1 * state_b0 + config_b2 * state_b1 -
                config_a0 * state_a0 - config_a1 * state_a1;
    state_b1 = state_b0;
    state_b0 = tmp;
    state_a1 = state_a0;
    state_a0 = y_k;
    y[k] = y_k;
  }
  state_.a[0] = state_a0;
  state_.a[1] = state_a1;
  state_.b[0] = state_b0;
  state_.b[1] = state_b1;
}

}  // namespace webrtc
