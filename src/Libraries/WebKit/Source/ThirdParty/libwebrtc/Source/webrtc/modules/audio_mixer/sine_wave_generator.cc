/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 16, 2022.
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
#include "modules/audio_mixer/sine_wave_generator.h"

#include <math.h>
#include <stddef.h>

#include "rtc_base/numerics/safe_conversions.h"

namespace webrtc {

namespace {
constexpr float kPi = 3.14159265f;
}  // namespace

void SineWaveGenerator::GenerateNextFrame(AudioFrame* frame) {
  RTC_DCHECK(frame);
  int16_t* frame_data = frame->mutable_data();
  for (size_t i = 0; i < frame->samples_per_channel_; ++i) {
    for (size_t ch = 0; ch < frame->num_channels_; ++ch) {
      frame_data[frame->num_channels_ * i + ch] =
          rtc::saturated_cast<int16_t>(amplitude_ * sinf(phase_));
    }
    phase_ += wave_frequency_hz_ * 2 * kPi / frame->sample_rate_hz_;
  }
}
}  // namespace webrtc
