/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 7, 2025.
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
#include "modules/audio_processing/aec3/reverb_model.h"

#include <stddef.h>

#include <algorithm>
#include <functional>

#include "api/array_view.h"

namespace webrtc {

ReverbModel::ReverbModel() {
  Reset();
}

ReverbModel::~ReverbModel() = default;

void ReverbModel::Reset() {
  reverb_.fill(0.);
}

void ReverbModel::UpdateReverbNoFreqShaping(
    rtc::ArrayView<const float> power_spectrum,
    float power_spectrum_scaling,
    float reverb_decay) {
  if (reverb_decay > 0) {
    // Update the estimate of the reverberant power.
    for (size_t k = 0; k < power_spectrum.size(); ++k) {
      reverb_[k] = (reverb_[k] + power_spectrum[k] * power_spectrum_scaling) *
                   reverb_decay;
    }
  }
}

void ReverbModel::UpdateReverb(
    rtc::ArrayView<const float> power_spectrum,
    rtc::ArrayView<const float> power_spectrum_scaling,
    float reverb_decay) {
  if (reverb_decay > 0) {
    // Update the estimate of the reverberant power.
    for (size_t k = 0; k < power_spectrum.size(); ++k) {
      reverb_[k] =
          (reverb_[k] + power_spectrum[k] * power_spectrum_scaling[k]) *
          reverb_decay;
    }
  }
}

}  // namespace webrtc
