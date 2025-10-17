/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 4, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_NS_WIENER_FILTER_H_
#define MODULES_AUDIO_PROCESSING_NS_WIENER_FILTER_H_

#include <array>

#include "api/array_view.h"
#include "modules/audio_processing/ns/ns_common.h"
#include "modules/audio_processing/ns/suppression_params.h"

namespace webrtc {

// Estimates a Wiener-filter based frequency domain noise reduction filter.
class WienerFilter {
 public:
  explicit WienerFilter(const SuppressionParams& suppression_params);
  WienerFilter(const WienerFilter&) = delete;
  WienerFilter& operator=(const WienerFilter&) = delete;

  // Updates the filter estimate.
  void Update(
      int32_t num_analyzed_frames,
      rtc::ArrayView<const float, kFftSizeBy2Plus1> noise_spectrum,
      rtc::ArrayView<const float, kFftSizeBy2Plus1> prev_noise_spectrum,
      rtc::ArrayView<const float, kFftSizeBy2Plus1> parametric_noise_spectrum,
      rtc::ArrayView<const float, kFftSizeBy2Plus1> signal_spectrum);

  // Compute an overall gain scaling factor.
  float ComputeOverallScalingFactor(int32_t num_analyzed_frames,
                                    float prior_speech_probability,
                                    float energy_before_filtering,
                                    float energy_after_filtering) const;

  // Returns the filter.
  rtc::ArrayView<const float, kFftSizeBy2Plus1> get_filter() const {
    return filter_;
  }

 private:
  const SuppressionParams& suppression_params_;
  std::array<float, kFftSizeBy2Plus1> spectrum_prev_process_;
  std::array<float, kFftSizeBy2Plus1> initial_spectral_estimate_;
  std::array<float, kFftSizeBy2Plus1> filter_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_NS_WIENER_FILTER_H_
