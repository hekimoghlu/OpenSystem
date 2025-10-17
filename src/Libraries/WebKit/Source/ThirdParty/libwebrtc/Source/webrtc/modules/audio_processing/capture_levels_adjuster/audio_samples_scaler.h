/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 16, 2025.
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
#ifndef MODULES_AUDIO_PROCESSING_CAPTURE_LEVELS_ADJUSTER_AUDIO_SAMPLES_SCALER_H_
#define MODULES_AUDIO_PROCESSING_CAPTURE_LEVELS_ADJUSTER_AUDIO_SAMPLES_SCALER_H_

#include <stddef.h>

#include "modules/audio_processing/audio_buffer.h"

namespace webrtc {

// Handles and applies a gain to the samples in an audio buffer.
// The gain is applied for each sample and any changes in the gain take effect
// gradually (in a linear manner) over one frame.
class AudioSamplesScaler {
 public:
  // C-tor. The supplied `initial_gain` is used immediately at the first call to
  // Process(), i.e., in contrast to the gain supplied by SetGain(...) there is
  // no gradual change to the `initial_gain`.
  explicit AudioSamplesScaler(float initial_gain);
  AudioSamplesScaler(const AudioSamplesScaler&) = delete;
  AudioSamplesScaler& operator=(const AudioSamplesScaler&) = delete;

  // Applies the specified gain to the audio in `audio_buffer`.
  void Process(AudioBuffer& audio_buffer);

  // Sets the gain to apply to each sample.
  void SetGain(float gain) { target_gain_ = gain; }

 private:
  float previous_gain_ = 1.f;
  float target_gain_ = 1.f;
  int samples_per_channel_ = -1;
  float one_by_samples_per_channel_ = -1.f;
};
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_CAPTURE_LEVELS_ADJUSTER_AUDIO_SAMPLES_SCALER_H_
