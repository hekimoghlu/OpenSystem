/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 22, 2025.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_DOMINANT_NEAREND_DETECTOR_H_
#define MODULES_AUDIO_PROCESSING_AEC3_DOMINANT_NEAREND_DETECTOR_H_

#include <vector>

#include "api/array_view.h"
#include "api/audio/echo_canceller3_config.h"
#include "modules/audio_processing/aec3/nearend_detector.h"

namespace webrtc {
// Class for selecting whether the suppressor is in the nearend or echo state.
class DominantNearendDetector : public NearendDetector {
 public:
  DominantNearendDetector(
      const EchoCanceller3Config::Suppressor::DominantNearendDetection& config,
      size_t num_capture_channels);

  // Returns whether the current state is the nearend state.
  bool IsNearendState() const override { return nearend_state_; }

  // Updates the state selection based on latest spectral estimates.
  void Update(rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>>
                  nearend_spectrum,
              rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>>
                  residual_echo_spectrum,
              rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>>
                  comfort_noise_spectrum,
              bool initial_state) override;

 private:
  const float enr_threshold_;
  const float enr_exit_threshold_;
  const float snr_threshold_;
  const int hold_duration_;
  const int trigger_threshold_;
  const bool use_during_initial_phase_;
  const size_t num_capture_channels_;

  bool nearend_state_ = false;
  std::vector<int> trigger_counters_;
  std::vector<int> hold_counters_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_DOMINANT_NEAREND_DETECTOR_H_
