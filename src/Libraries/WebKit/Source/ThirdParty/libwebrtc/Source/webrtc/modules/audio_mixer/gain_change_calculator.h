/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 24, 2024.
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
#ifndef MODULES_AUDIO_MIXER_GAIN_CHANGE_CALCULATOR_H_
#define MODULES_AUDIO_MIXER_GAIN_CHANGE_CALCULATOR_H_

#include <stdint.h>

#include "api/array_view.h"

namespace webrtc {

class GainChangeCalculator {
 public:
  // The 'out' signal is assumed to be produced from 'in' by applying
  // a smoothly varying gain. This method computes variations of the
  // gain and handles special cases when the samples are small.
  float CalculateGainChange(rtc::ArrayView<const int16_t> in,
                            rtc::ArrayView<const int16_t> out);

  float LatestGain() const;

 private:
  void CalculateGain(rtc::ArrayView<const int16_t> in,
                     rtc::ArrayView<const int16_t> out,
                     rtc::ArrayView<float> gain);

  float CalculateDifferences(rtc::ArrayView<const float> values);
  float last_value_ = 0.f;
  float last_reliable_gain_ = 1.0f;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_MIXER_GAIN_CHANGE_CALCULATOR_H_
