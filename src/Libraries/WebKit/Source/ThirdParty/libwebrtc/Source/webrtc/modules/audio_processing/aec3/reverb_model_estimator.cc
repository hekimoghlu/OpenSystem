/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 28, 2024.
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
#include "modules/audio_processing/aec3/reverb_model_estimator.h"

namespace webrtc {

ReverbModelEstimator::ReverbModelEstimator(const EchoCanceller3Config& config,
                                           size_t num_capture_channels)
    : reverb_decay_estimators_(num_capture_channels),
      reverb_frequency_responses_(
          num_capture_channels,
          ReverbFrequencyResponse(
              config.ep_strength.use_conservative_tail_frequency_response)) {
  for (size_t ch = 0; ch < reverb_decay_estimators_.size(); ++ch) {
    reverb_decay_estimators_[ch] =
        std::make_unique<ReverbDecayEstimator>(config);
  }
}

ReverbModelEstimator::~ReverbModelEstimator() = default;

void ReverbModelEstimator::Update(
    rtc::ArrayView<const std::vector<float>> impulse_responses,
    rtc::ArrayView<const std::vector<std::array<float, kFftLengthBy2Plus1>>>
        frequency_responses,
    rtc::ArrayView<const std::optional<float>> linear_filter_qualities,
    rtc::ArrayView<const int> filter_delays_blocks,
    const std::vector<bool>& usable_linear_estimates,
    bool stationary_block) {
  const size_t num_capture_channels = reverb_decay_estimators_.size();
  RTC_DCHECK_EQ(num_capture_channels, impulse_responses.size());
  RTC_DCHECK_EQ(num_capture_channels, frequency_responses.size());
  RTC_DCHECK_EQ(num_capture_channels, usable_linear_estimates.size());

  for (size_t ch = 0; ch < num_capture_channels; ++ch) {
    // Estimate the frequency response for the reverb.
    reverb_frequency_responses_[ch].Update(
        frequency_responses[ch], filter_delays_blocks[ch],
        linear_filter_qualities[ch], stationary_block);

    // Estimate the reverb decay,
    reverb_decay_estimators_[ch]->Update(
        impulse_responses[ch], linear_filter_qualities[ch],
        filter_delays_blocks[ch], usable_linear_estimates[ch],
        stationary_block);
  }
}

}  // namespace webrtc
