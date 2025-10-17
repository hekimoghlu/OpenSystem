/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 2, 2021.
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
#ifndef MODULES_AUDIO_PROCESSING_NS_SPEECH_PROBABILITY_ESTIMATOR_H_
#define MODULES_AUDIO_PROCESSING_NS_SPEECH_PROBABILITY_ESTIMATOR_H_

#include <array>

#include "api/array_view.h"
#include "modules/audio_processing/ns/ns_common.h"
#include "modules/audio_processing/ns/signal_model_estimator.h"

namespace webrtc {

// Class for estimating the probability of speech.
class SpeechProbabilityEstimator {
 public:
  SpeechProbabilityEstimator();
  SpeechProbabilityEstimator(const SpeechProbabilityEstimator&) = delete;
  SpeechProbabilityEstimator& operator=(const SpeechProbabilityEstimator&) =
      delete;

  // Compute speech probability.
  void Update(
      int32_t num_analyzed_frames,
      rtc::ArrayView<const float, kFftSizeBy2Plus1> prior_snr,
      rtc::ArrayView<const float, kFftSizeBy2Plus1> post_snr,
      rtc::ArrayView<const float, kFftSizeBy2Plus1> conservative_noise_spectrum,
      rtc::ArrayView<const float, kFftSizeBy2Plus1> signal_spectrum,
      float signal_spectral_sum,
      float signal_energy);

  float get_prior_probability() const { return prior_speech_prob_; }
  rtc::ArrayView<const float> get_probability() { return speech_probability_; }

 private:
  SignalModelEstimator signal_model_estimator_;
  float prior_speech_prob_ = .5f;
  std::array<float, kFftSizeBy2Plus1> speech_probability_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_NS_SPEECH_PROBABILITY_ESTIMATOR_H_
