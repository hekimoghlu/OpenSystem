/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 7, 2025.
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
#ifndef MODULES_AUDIO_PROCESSING_NS_SIGNAL_MODEL_ESTIMATOR_H_
#define MODULES_AUDIO_PROCESSING_NS_SIGNAL_MODEL_ESTIMATOR_H_

#include <array>

#include "api/array_view.h"
#include "modules/audio_processing/ns/histograms.h"
#include "modules/audio_processing/ns/ns_common.h"
#include "modules/audio_processing/ns/prior_signal_model.h"
#include "modules/audio_processing/ns/prior_signal_model_estimator.h"
#include "modules/audio_processing/ns/signal_model.h"

namespace webrtc {

class SignalModelEstimator {
 public:
  SignalModelEstimator();
  SignalModelEstimator(const SignalModelEstimator&) = delete;
  SignalModelEstimator& operator=(const SignalModelEstimator&) = delete;

  // Compute signal normalization during the initial startup phase.
  void AdjustNormalization(int32_t num_analyzed_frames, float signal_energy);

  void Update(
      rtc::ArrayView<const float, kFftSizeBy2Plus1> prior_snr,
      rtc::ArrayView<const float, kFftSizeBy2Plus1> post_snr,
      rtc::ArrayView<const float, kFftSizeBy2Plus1> conservative_noise_spectrum,
      rtc::ArrayView<const float, kFftSizeBy2Plus1> signal_spectrum,
      float signal_spectral_sum,
      float signal_energy);

  const PriorSignalModel& get_prior_model() const {
    return prior_model_estimator_.get_prior_model();
  }
  const SignalModel& get_model() { return features_; }

 private:
  float diff_normalization_ = 0.f;
  float signal_energy_sum_ = 0.f;
  Histograms histograms_;
  int histogram_analysis_counter_ = 500;
  PriorSignalModelEstimator prior_model_estimator_;
  SignalModel features_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_NS_SIGNAL_MODEL_ESTIMATOR_H_
