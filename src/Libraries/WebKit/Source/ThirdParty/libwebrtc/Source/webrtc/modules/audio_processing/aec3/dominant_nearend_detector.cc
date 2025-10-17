/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 19, 2022.
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
#include "modules/audio_processing/aec3/dominant_nearend_detector.h"

#include <numeric>

namespace webrtc {
DominantNearendDetector::DominantNearendDetector(
    const EchoCanceller3Config::Suppressor::DominantNearendDetection& config,
    size_t num_capture_channels)
    : enr_threshold_(config.enr_threshold),
      enr_exit_threshold_(config.enr_exit_threshold),
      snr_threshold_(config.snr_threshold),
      hold_duration_(config.hold_duration),
      trigger_threshold_(config.trigger_threshold),
      use_during_initial_phase_(config.use_during_initial_phase),
      num_capture_channels_(num_capture_channels),
      trigger_counters_(num_capture_channels_),
      hold_counters_(num_capture_channels_) {}

void DominantNearendDetector::Update(
    rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>>
        nearend_spectrum,
    rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>>
        residual_echo_spectrum,
    rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>>
        comfort_noise_spectrum,
    bool initial_state) {
  nearend_state_ = false;

  auto low_frequency_energy = [](rtc::ArrayView<const float> spectrum) {
    RTC_DCHECK_LE(16, spectrum.size());
    return std::accumulate(spectrum.begin() + 1, spectrum.begin() + 16, 0.f);
  };

  for (size_t ch = 0; ch < num_capture_channels_; ++ch) {
    const float ne_sum = low_frequency_energy(nearend_spectrum[ch]);
    const float echo_sum = low_frequency_energy(residual_echo_spectrum[ch]);
    const float noise_sum = low_frequency_energy(comfort_noise_spectrum[ch]);

    // Detect strong active nearend if the nearend is sufficiently stronger than
    // the echo and the nearend noise.
    if ((!initial_state || use_during_initial_phase_) &&
        echo_sum < enr_threshold_ * ne_sum &&
        ne_sum > snr_threshold_ * noise_sum) {
      if (++trigger_counters_[ch] >= trigger_threshold_) {
        // After a period of strong active nearend activity, flag nearend mode.
        hold_counters_[ch] = hold_duration_;
        trigger_counters_[ch] = trigger_threshold_;
      }
    } else {
      // Forget previously detected strong active nearend activity.
      trigger_counters_[ch] = std::max(0, trigger_counters_[ch] - 1);
    }

    // Exit nearend-state early at strong echo.
    if (echo_sum > enr_exit_threshold_ * ne_sum &&
        echo_sum > snr_threshold_ * noise_sum) {
      hold_counters_[ch] = 0;
    }

    // Remain in any nearend mode for a certain duration.
    hold_counters_[ch] = std::max(0, hold_counters_[ch] - 1);
    nearend_state_ = nearend_state_ || hold_counters_[ch] > 0;
  }
}
}  // namespace webrtc
