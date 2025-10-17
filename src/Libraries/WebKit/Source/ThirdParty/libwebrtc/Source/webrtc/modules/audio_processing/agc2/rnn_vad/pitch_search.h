/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 2, 2025.
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
#ifndef MODULES_AUDIO_PROCESSING_AGC2_RNN_VAD_PITCH_SEARCH_H_
#define MODULES_AUDIO_PROCESSING_AGC2_RNN_VAD_PITCH_SEARCH_H_

#include <memory>
#include <vector>

#include "api/array_view.h"
#include "modules/audio_processing/agc2/cpu_features.h"
#include "modules/audio_processing/agc2/rnn_vad/auto_correlation.h"
#include "modules/audio_processing/agc2/rnn_vad/common.h"
#include "modules/audio_processing/agc2/rnn_vad/pitch_search_internal.h"
#include "rtc_base/gtest_prod_util.h"

namespace webrtc {
namespace rnn_vad {

// Pitch estimator.
class PitchEstimator {
 public:
  explicit PitchEstimator(const AvailableCpuFeatures& cpu_features);
  PitchEstimator(const PitchEstimator&) = delete;
  PitchEstimator& operator=(const PitchEstimator&) = delete;
  ~PitchEstimator();
  // Returns the estimated pitch period at 48 kHz.
  int Estimate(rtc::ArrayView<const float, kBufSize24kHz> pitch_buffer);

 private:
  FRIEND_TEST_ALL_PREFIXES(RnnVadTest, PitchSearchWithinTolerance);
  float GetLastPitchStrengthForTesting() const {
    return last_pitch_48kHz_.strength;
  }

  const AvailableCpuFeatures cpu_features_;
  PitchInfo last_pitch_48kHz_{};
  AutoCorrelationCalculator auto_corr_calculator_;
  std::vector<float> y_energy_24kHz_;
  std::vector<float> pitch_buffer_12kHz_;
  std::vector<float> auto_correlation_12kHz_;
};

}  // namespace rnn_vad
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AGC2_RNN_VAD_PITCH_SEARCH_H_
