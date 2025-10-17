/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 27, 2024.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_COARSE_FILTER_UPDATE_GAIN_H_
#define MODULES_AUDIO_PROCESSING_AEC3_COARSE_FILTER_UPDATE_GAIN_H_

#include <stddef.h>

#include <array>

#include "api/audio/echo_canceller3_config.h"
#include "modules/audio_processing/aec3/aec3_common.h"
#include "modules/audio_processing/aec3/fft_data.h"
#include "modules/audio_processing/aec3/render_signal_analyzer.h"

namespace webrtc {

// Provides functionality for computing the fixed gain for the coarse filter.
class CoarseFilterUpdateGain {
 public:
  explicit CoarseFilterUpdateGain(
      const EchoCanceller3Config::Filter::CoarseConfiguration& config,
      size_t config_change_duration_blocks);

  // Takes action in the case of a known echo path change.
  void HandleEchoPathChange();

  // Computes the gain.
  void Compute(const std::array<float, kFftLengthBy2Plus1>& render_power,
               const RenderSignalAnalyzer& render_signal_analyzer,
               const FftData& E_coarse,
               size_t size_partitions,
               bool saturated_capture_signal,
               FftData* G);

  // Sets a new config.
  void SetConfig(
      const EchoCanceller3Config::Filter::CoarseConfiguration& config,
      bool immediate_effect) {
    if (immediate_effect) {
      old_target_config_ = current_config_ = target_config_ = config;
      config_change_counter_ = 0;
    } else {
      old_target_config_ = current_config_;
      target_config_ = config;
      config_change_counter_ = config_change_duration_blocks_;
    }
  }

 private:
  EchoCanceller3Config::Filter::CoarseConfiguration current_config_;
  EchoCanceller3Config::Filter::CoarseConfiguration target_config_;
  EchoCanceller3Config::Filter::CoarseConfiguration old_target_config_;
  const int config_change_duration_blocks_;
  float one_by_config_change_duration_blocks_;
  // TODO(peah): Check whether this counter should instead be initialized to a
  // large value.
  size_t poor_signal_excitation_counter_ = 0;
  size_t call_counter_ = 0;
  int config_change_counter_ = 0;

  void UpdateCurrentConfig();
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_COARSE_FILTER_UPDATE_GAIN_H_
