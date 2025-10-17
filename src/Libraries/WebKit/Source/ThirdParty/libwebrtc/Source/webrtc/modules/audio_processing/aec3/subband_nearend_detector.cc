/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 6, 2025.
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
#include "modules/audio_processing/aec3/subband_nearend_detector.h"

#include <numeric>

namespace webrtc {
SubbandNearendDetector::SubbandNearendDetector(
    const EchoCanceller3Config::Suppressor::SubbandNearendDetection& config,
    size_t num_capture_channels)
    : config_(config),
      num_capture_channels_(num_capture_channels),
      nearend_smoothers_(num_capture_channels_,
                         aec3::MovingAverage(kFftLengthBy2Plus1,
                                             config_.nearend_average_blocks)),
      one_over_subband_length1_(
          1.f / (config_.subband1.high - config_.subband1.low + 1)),
      one_over_subband_length2_(
          1.f / (config_.subband2.high - config_.subband2.low + 1)) {}

void SubbandNearendDetector::Update(
    rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>>
        nearend_spectrum,
    rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>>
    /* residual_echo_spectrum */,
    rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>>
        comfort_noise_spectrum,
    bool /* initial_state */) {
  nearend_state_ = false;
  for (size_t ch = 0; ch < num_capture_channels_; ++ch) {
    const std::array<float, kFftLengthBy2Plus1>& noise =
        comfort_noise_spectrum[ch];
    std::array<float, kFftLengthBy2Plus1> nearend;
    nearend_smoothers_[ch].Average(nearend_spectrum[ch], nearend);

    // Noise power of the first region.
    float noise_power =
        std::accumulate(noise.begin() + config_.subband1.low,
                        noise.begin() + config_.subband1.high + 1, 0.f) *
        one_over_subband_length1_;

    // Nearend power of the first region.
    float nearend_power_subband1 =
        std::accumulate(nearend.begin() + config_.subband1.low,
                        nearend.begin() + config_.subband1.high + 1, 0.f) *
        one_over_subband_length1_;

    // Nearend power of the second region.
    float nearend_power_subband2 =
        std::accumulate(nearend.begin() + config_.subband2.low,
                        nearend.begin() + config_.subband2.high + 1, 0.f) *
        one_over_subband_length2_;

    // One channel is sufficient to trigger nearend state.
    nearend_state_ =
        nearend_state_ ||
        (nearend_power_subband1 <
             config_.nearend_threshold * nearend_power_subband2 &&
         (nearend_power_subband1 > config_.snr_threshold * noise_power));
  }
}
}  // namespace webrtc
