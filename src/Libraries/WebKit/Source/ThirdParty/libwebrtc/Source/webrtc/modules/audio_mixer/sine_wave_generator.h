/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 16, 2024.
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
#ifndef MODULES_AUDIO_MIXER_SINE_WAVE_GENERATOR_H_
#define MODULES_AUDIO_MIXER_SINE_WAVE_GENERATOR_H_

#include <stdint.h>

#include "api/audio/audio_frame.h"
#include "rtc_base/checks.h"

namespace webrtc {

class SineWaveGenerator {
 public:
  SineWaveGenerator(float wave_frequency_hz, int16_t amplitude)
      : wave_frequency_hz_(wave_frequency_hz), amplitude_(amplitude) {
    RTC_DCHECK_GT(wave_frequency_hz, 0);
  }

  // Produces appropriate output based on frame->num_channels_,
  // frame->sample_rate_hz_.
  void GenerateNextFrame(AudioFrame* frame);

 private:
  float phase_ = 0.f;
  const float wave_frequency_hz_;
  const int16_t amplitude_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_MIXER_SINE_WAVE_GENERATOR_H_
