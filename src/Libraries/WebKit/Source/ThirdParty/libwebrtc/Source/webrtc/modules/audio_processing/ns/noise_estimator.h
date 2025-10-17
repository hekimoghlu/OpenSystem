/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 16, 2021.
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
#ifndef MODULES_AUDIO_PROCESSING_NS_NOISE_ESTIMATOR_H_
#define MODULES_AUDIO_PROCESSING_NS_NOISE_ESTIMATOR_H_

#include <array>

#include "api/array_view.h"
#include "modules/audio_processing/ns/ns_common.h"
#include "modules/audio_processing/ns/quantile_noise_estimator.h"
#include "modules/audio_processing/ns/suppression_params.h"

namespace webrtc {

// Class for estimating the spectral characteristics of the noise in an incoming
// signal.
class NoiseEstimator {
 public:
  explicit NoiseEstimator(const SuppressionParams& suppression_params);

  // Prepare the estimator for analysis of a new frame.
  void PrepareAnalysis();

  // Performs the first step of the estimator update.
  void PreUpdate(int32_t num_analyzed_frames,
                 rtc::ArrayView<const float, kFftSizeBy2Plus1> signal_spectrum,
                 float signal_spectral_sum);

  // Performs the second step of the estimator update.
  void PostUpdate(
      rtc::ArrayView<const float> speech_probability,
      rtc::ArrayView<const float, kFftSizeBy2Plus1> signal_spectrum);

  // Returns the noise spectral estimate.
  rtc::ArrayView<const float, kFftSizeBy2Plus1> get_noise_spectrum() const {
    return noise_spectrum_;
  }

  // Returns the noise from the previous frame.
  rtc::ArrayView<const float, kFftSizeBy2Plus1> get_prev_noise_spectrum()
      const {
    return prev_noise_spectrum_;
  }

  // Returns a noise spectral estimate based on white and pink noise parameters.
  rtc::ArrayView<const float, kFftSizeBy2Plus1> get_parametric_noise_spectrum()
      const {
    return parametric_noise_spectrum_;
  }
  rtc::ArrayView<const float, kFftSizeBy2Plus1>
  get_conservative_noise_spectrum() const {
    return conservative_noise_spectrum_;
  }

 private:
  const SuppressionParams& suppression_params_;
  float white_noise_level_ = 0.f;
  float pink_noise_numerator_ = 0.f;
  float pink_noise_exp_ = 0.f;
  std::array<float, kFftSizeBy2Plus1> prev_noise_spectrum_;
  std::array<float, kFftSizeBy2Plus1> conservative_noise_spectrum_;
  std::array<float, kFftSizeBy2Plus1> parametric_noise_spectrum_;
  std::array<float, kFftSizeBy2Plus1> noise_spectrum_;
  QuantileNoiseEstimator quantile_noise_estimator_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_NS_NOISE_ESTIMATOR_H_
