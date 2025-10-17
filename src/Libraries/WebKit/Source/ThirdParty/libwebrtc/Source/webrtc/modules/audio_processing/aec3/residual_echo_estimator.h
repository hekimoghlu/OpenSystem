/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 19, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_RESIDUAL_ECHO_ESTIMATOR_H_
#define MODULES_AUDIO_PROCESSING_AEC3_RESIDUAL_ECHO_ESTIMATOR_H_

#include <array>
#include <memory>
#include <optional>

#include "api/audio/echo_canceller3_config.h"
#include "modules/audio_processing/aec3/aec3_common.h"
#include "modules/audio_processing/aec3/aec_state.h"
#include "modules/audio_processing/aec3/render_buffer.h"
#include "modules/audio_processing/aec3/reverb_model.h"
#include "modules/audio_processing/aec3/spectrum_buffer.h"
#include "rtc_base/checks.h"

namespace webrtc {

class ResidualEchoEstimator {
 public:
  ResidualEchoEstimator(const EchoCanceller3Config& config,
                        size_t num_render_channels);
  ~ResidualEchoEstimator();

  ResidualEchoEstimator(const ResidualEchoEstimator&) = delete;
  ResidualEchoEstimator& operator=(const ResidualEchoEstimator&) = delete;

  void Estimate(
      const AecState& aec_state,
      const RenderBuffer& render_buffer,
      rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>> S2_linear,
      rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>> Y2,
      bool dominant_nearend,
      rtc::ArrayView<std::array<float, kFftLengthBy2Plus1>> R2,
      rtc::ArrayView<std::array<float, kFftLengthBy2Plus1>> R2_unbounded);

 private:
  enum class ReverbType { kLinear, kNonLinear };

  // Resets the state.
  void Reset();

  // Updates estimate for the power of the stationary noise component in the
  // render signal.
  void UpdateRenderNoisePower(const RenderBuffer& render_buffer);

  // Updates the reverb estimation.
  void UpdateReverb(ReverbType reverb_type,
                    const AecState& aec_state,
                    const RenderBuffer& render_buffer,
                    bool dominant_nearend);

  // Adds the estimated unmodelled echo power to the residual echo power
  // estimate.
  void AddReverb(
      rtc::ArrayView<std::array<float, kFftLengthBy2Plus1>> R2) const;

  // Gets the echo path gain to apply.
  float GetEchoPathGain(const AecState& aec_state,
                        bool gain_for_early_reflections) const;

  const EchoCanceller3Config config_;
  const size_t num_render_channels_;
  const float early_reflections_transparent_mode_gain_;
  const float late_reflections_transparent_mode_gain_;
  const float early_reflections_general_gain_;
  const float late_reflections_general_gain_;
  const bool erle_onset_compensation_in_dominant_nearend_;
  std::array<float, kFftLengthBy2Plus1> X2_noise_floor_;
  std::array<int, kFftLengthBy2Plus1> X2_noise_floor_counter_;
  ReverbModel echo_reverb_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_RESIDUAL_ECHO_ESTIMATOR_H_
