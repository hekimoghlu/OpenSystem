/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 7, 2022.
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
#ifndef MODULES_AUDIO_PROCESSING_AGC2_GAIN_APPLIER_H_
#define MODULES_AUDIO_PROCESSING_AGC2_GAIN_APPLIER_H_

#include <stddef.h>

#include "api/audio/audio_view.h"
#include "modules/audio_processing/include/audio_frame_view.h"

namespace webrtc {
class GainApplier {
 public:
  GainApplier(bool hard_clip_samples, float initial_gain_factor);

  void ApplyGain(DeinterleavedView<float> signal);
  void SetGainFactor(float gain_factor);
  float GetGainFactor() const { return current_gain_factor_; }

  [[deprecated("Use DeinterleavedView<> version")]] void ApplyGain(
      AudioFrameView<float> signal) {
    ApplyGain(signal.view());
  }

 private:
  void Initialize(int samples_per_channel);

  // Whether to clip samples after gain is applied. If 'true', result
  // will fit in FloatS16 range.
  const bool hard_clip_samples_;
  float last_gain_factor_;

  // If this value is not equal to 'last_gain_factor', gain will be
  // ramped from 'last_gain_factor_' to this value during the next
  // 'ApplyGain'.
  float current_gain_factor_;
  int samples_per_channel_ = -1;
  float inverse_samples_per_channel_ = -1.f;
};
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AGC2_GAIN_APPLIER_H_
