/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 6, 2023.
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
#include "modules/audio_processing/ns/speech_probability_estimator.h"

#include <math.h>

#include <algorithm>

#include "modules/audio_processing/ns/fast_math.h"
#include "rtc_base/checks.h"

namespace webrtc {

SpeechProbabilityEstimator::SpeechProbabilityEstimator() {
  speech_probability_.fill(0.f);
}

void SpeechProbabilityEstimator::Update(
    int32_t num_analyzed_frames,
    rtc::ArrayView<const float, kFftSizeBy2Plus1> prior_snr,
    rtc::ArrayView<const float, kFftSizeBy2Plus1> post_snr,
    rtc::ArrayView<const float, kFftSizeBy2Plus1> conservative_noise_spectrum,
    rtc::ArrayView<const float, kFftSizeBy2Plus1> signal_spectrum,
    float signal_spectral_sum,
    float signal_energy) {
  // Update models.
  if (num_analyzed_frames < kLongStartupPhaseBlocks) {
    signal_model_estimator_.AdjustNormalization(num_analyzed_frames,
                                                signal_energy);
  }
  signal_model_estimator_.Update(prior_snr, post_snr,
                                 conservative_noise_spectrum, signal_spectrum,
                                 signal_spectral_sum, signal_energy);

  const SignalModel& model = signal_model_estimator_.get_model();
  const PriorSignalModel& prior_model =
      signal_model_estimator_.get_prior_model();

  // Width parameter in sigmoid map for prior model.
  constexpr float kWidthPrior0 = 4.f;
  // Width for pause region: lower range, so increase width in tanh map.
  constexpr float kWidthPrior1 = 2.f * kWidthPrior0;

  // Average LRT feature: use larger width in tanh map for pause regions.
  float width_prior = model.lrt < prior_model.lrt ? kWidthPrior1 : kWidthPrior0;

  // Compute indicator function: sigmoid map.
  float indicator0 =
      0.5f * (tanh(width_prior * (model.lrt - prior_model.lrt)) + 1.f);

  // Spectral flatness feature: use larger width in tanh map for pause regions.
  width_prior = model.spectral_flatness > prior_model.flatness_threshold
                    ? kWidthPrior1
                    : kWidthPrior0;

  // Compute indicator function: sigmoid map.
  float indicator1 =
      0.5f * (tanh(1.f * width_prior *
                   (prior_model.flatness_threshold - model.spectral_flatness)) +
              1.f);

  // For template spectrum-difference : use larger width in tanh map for pause
  // regions.
  width_prior = model.spectral_diff < prior_model.template_diff_threshold
                    ? kWidthPrior1
                    : kWidthPrior0;

  // Compute indicator function: sigmoid map.
  float indicator2 =
      0.5f * (tanh(width_prior * (model.spectral_diff -
                                  prior_model.template_diff_threshold)) +
              1.f);

  // Combine the indicator function with the feature weights.
  float ind_prior = prior_model.lrt_weighting * indicator0 +
                    prior_model.flatness_weighting * indicator1 +
                    prior_model.difference_weighting * indicator2;

  // Compute the prior probability.
  prior_speech_prob_ += 0.1f * (ind_prior - prior_speech_prob_);

  // Make sure probabilities are within range: keep floor to 0.01.
  prior_speech_prob_ = std::max(std::min(prior_speech_prob_, 1.f), 0.01f);

  // Final speech probability: combine prior model with LR factor:.
  float gain_prior =
      (1.f - prior_speech_prob_) / (prior_speech_prob_ + 0.0001f);

  std::array<float, kFftSizeBy2Plus1> inv_lrt;
  ExpApproximationSignFlip(model.avg_log_lrt, inv_lrt);
  for (size_t i = 0; i < kFftSizeBy2Plus1; ++i) {
    speech_probability_[i] = 1.f / (1.f + gain_prior * inv_lrt[i]);
  }
}

}  // namespace webrtc
