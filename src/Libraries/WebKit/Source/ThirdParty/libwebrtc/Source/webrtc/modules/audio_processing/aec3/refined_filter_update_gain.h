/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 4, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_REFINED_FILTER_UPDATE_GAIN_H_
#define MODULES_AUDIO_PROCESSING_AEC3_REFINED_FILTER_UPDATE_GAIN_H_

#include <stddef.h>

#include <array>
#include <atomic>
#include <memory>

#include "api/array_view.h"
#include "api/audio/echo_canceller3_config.h"
#include "modules/audio_processing/aec3/aec3_common.h"

namespace webrtc {

class AdaptiveFirFilter;
class ApmDataDumper;
struct EchoPathVariability;
struct FftData;
class RenderSignalAnalyzer;
struct SubtractorOutput;

// Provides functionality for  computing the adaptive gain for the refined
// filter.
class RefinedFilterUpdateGain {
 public:
  RefinedFilterUpdateGain(
      const EchoCanceller3Config::Filter::RefinedConfiguration& config,
      size_t config_change_duration_blocks);
  ~RefinedFilterUpdateGain();

  RefinedFilterUpdateGain(const RefinedFilterUpdateGain&) = delete;
  RefinedFilterUpdateGain& operator=(const RefinedFilterUpdateGain&) = delete;

  // Takes action in the case of a known echo path change.
  void HandleEchoPathChange(const EchoPathVariability& echo_path_variability);

  // Computes the gain.
  void Compute(const std::array<float, kFftLengthBy2Plus1>& render_power,
               const RenderSignalAnalyzer& render_signal_analyzer,
               const SubtractorOutput& subtractor_output,
               rtc::ArrayView<const float> erl,
               size_t size_partitions,
               bool saturated_capture_signal,
               bool disallow_leakage_diverged,
               FftData* gain_fft);

  // Sets a new config.
  void SetConfig(
      const EchoCanceller3Config::Filter::RefinedConfiguration& config,
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
  static std::atomic<int> instance_count_;
  std::unique_ptr<ApmDataDumper> data_dumper_;
  const int config_change_duration_blocks_;
  float one_by_config_change_duration_blocks_;
  EchoCanceller3Config::Filter::RefinedConfiguration current_config_;
  EchoCanceller3Config::Filter::RefinedConfiguration target_config_;
  EchoCanceller3Config::Filter::RefinedConfiguration old_target_config_;
  std::array<float, kFftLengthBy2Plus1> H_error_;
  size_t poor_excitation_counter_;
  size_t call_counter_ = 0;
  int config_change_counter_ = 0;

  // Updates the current config towards the target config.
  void UpdateCurrentConfig();
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_REFINED_FILTER_UPDATE_GAIN_H_
