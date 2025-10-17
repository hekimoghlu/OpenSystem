/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 21, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_REVERB_MODEL_H_
#define MODULES_AUDIO_PROCESSING_AEC3_REVERB_MODEL_H_

#include <array>

#include "api/array_view.h"
#include "modules/audio_processing/aec3/aec3_common.h"

namespace webrtc {

// The ReverbModel class describes an exponential reverberant model
// that can be applied over power spectrums.
class ReverbModel {
 public:
  ReverbModel();
  ~ReverbModel();

  // Resets the state.
  void Reset();

  // Returns the reverb.
  rtc::ArrayView<const float, kFftLengthBy2Plus1> reverb() const {
    return reverb_;
  }

  // The methods UpdateReverbNoFreqShaping and UpdateReverb update the
  // estimate of the reverberation contribution to an input/output power
  // spectrum. Before applying the exponential reverberant model, the input
  // power spectrum is pre-scaled. Use the method UpdateReverb when a different
  // scaling should be applied per frequency and UpdateReverb_no_freq_shape if
  // the same scaling should be used for all the frequencies.
  void UpdateReverbNoFreqShaping(rtc::ArrayView<const float> power_spectrum,
                                 float power_spectrum_scaling,
                                 float reverb_decay);

  // Update the reverb based on new data.
  void UpdateReverb(rtc::ArrayView<const float> power_spectrum,
                    rtc::ArrayView<const float> power_spectrum_scaling,
                    float reverb_decay);

 private:
  std::array<float, kFftLengthBy2Plus1> reverb_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_REVERB_MODEL_H_
