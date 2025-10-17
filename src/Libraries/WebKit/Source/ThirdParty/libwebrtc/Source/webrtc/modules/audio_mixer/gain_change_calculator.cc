/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 13, 2025.
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
#include "modules/audio_mixer/gain_change_calculator.h"

#include <math.h>

#include <cstdlib>
#include <vector>

#include "rtc_base/checks.h"

namespace webrtc {

namespace {
constexpr int16_t kReliabilityThreshold = 100;
}  // namespace

float GainChangeCalculator::CalculateGainChange(
    rtc::ArrayView<const int16_t> in,
    rtc::ArrayView<const int16_t> out) {
  RTC_DCHECK_EQ(in.size(), out.size());

  std::vector<float> gain(in.size());
  CalculateGain(in, out, gain);
  return CalculateDifferences(gain);
}

float GainChangeCalculator::LatestGain() const {
  return last_reliable_gain_;
}

void GainChangeCalculator::CalculateGain(rtc::ArrayView<const int16_t> in,
                                         rtc::ArrayView<const int16_t> out,
                                         rtc::ArrayView<float> gain) {
  RTC_DCHECK_EQ(in.size(), out.size());
  RTC_DCHECK_EQ(in.size(), gain.size());

  for (size_t i = 0; i < in.size(); ++i) {
    if (std::abs(in[i]) >= kReliabilityThreshold) {
      last_reliable_gain_ = out[i] / static_cast<float>(in[i]);
    }
    gain[i] = last_reliable_gain_;
  }
}

float GainChangeCalculator::CalculateDifferences(
    rtc::ArrayView<const float> values) {
  float res = 0;
  for (float f : values) {
    res += fabs(f - last_value_);
    last_value_ = f;
  }
  return res;
}
}  // namespace webrtc
