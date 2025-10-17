/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 9, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_SIGNAL_DEPENDENT_ERLE_ESTIMATOR_H_
#define MODULES_AUDIO_PROCESSING_AEC3_SIGNAL_DEPENDENT_ERLE_ESTIMATOR_H_

#include <memory>
#include <vector>

#include "api/array_view.h"
#include "api/audio/echo_canceller3_config.h"
#include "modules/audio_processing/aec3/aec3_common.h"
#include "modules/audio_processing/aec3/render_buffer.h"
#include "modules/audio_processing/logging/apm_data_dumper.h"

namespace webrtc {

// This class estimates the dependency of the Erle to the input signal. By
// looking at the input signal, an estimation on whether the current echo
// estimate is due to the direct path or to a more reverberant one is performed.
// Once that estimation is done, it is possible to refine the average Erle that
// this class receive as an input.
class SignalDependentErleEstimator {
 public:
  SignalDependentErleEstimator(const EchoCanceller3Config& config,
                               size_t num_capture_channels);

  ~SignalDependentErleEstimator();

  void Reset();

  // Returns the Erle per frequency subband.
  rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>> Erle(
      bool onset_compensated) const {
    return onset_compensated && use_onset_detection_ ? erle_onset_compensated_
                                                     : erle_;
  }

  // Updates the Erle estimate. The Erle that is passed as an input is required
  // to be an estimation of the average Erle achieved by the linear filter.
  void Update(
      const RenderBuffer& render_buffer,
      rtc::ArrayView<const std::vector<std::array<float, kFftLengthBy2Plus1>>>
          filter_frequency_response,
      rtc::ArrayView<const float, kFftLengthBy2Plus1> X2,
      rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>> Y2,
      rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>> E2,
      rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>> average_erle,
      rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>>
          average_erle_onset_compensated,
      const std::vector<bool>& converged_filters);

  void Dump(const std::unique_ptr<ApmDataDumper>& data_dumper) const;

  static constexpr size_t kSubbands = 6;

 private:
  void ComputeNumberOfActiveFilterSections(
      const RenderBuffer& render_buffer,
      rtc::ArrayView<const std::vector<std::array<float, kFftLengthBy2Plus1>>>
          filter_frequency_responses);

  void UpdateCorrectionFactors(
      rtc::ArrayView<const float, kFftLengthBy2Plus1> X2,
      rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>> Y2,
      rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>> E2,
      const std::vector<bool>& converged_filters);

  void ComputeEchoEstimatePerFilterSection(
      const RenderBuffer& render_buffer,
      rtc::ArrayView<const std::vector<std::array<float, kFftLengthBy2Plus1>>>
          filter_frequency_responses);

  void ComputeActiveFilterSections();

  const float min_erle_;
  const size_t num_sections_;
  const size_t num_blocks_;
  const size_t delay_headroom_blocks_;
  const std::array<size_t, kFftLengthBy2Plus1> band_to_subband_;
  const std::array<float, kSubbands> max_erle_;
  const std::vector<size_t> section_boundaries_blocks_;
  const bool use_onset_detection_;
  std::vector<std::array<float, kFftLengthBy2Plus1>> erle_;
  std::vector<std::array<float, kFftLengthBy2Plus1>> erle_onset_compensated_;
  std::vector<std::vector<std::array<float, kFftLengthBy2Plus1>>>
      S2_section_accum_;
  std::vector<std::vector<std::array<float, kSubbands>>> erle_estimators_;
  std::vector<std::array<float, kSubbands>> erle_ref_;
  std::vector<std::vector<std::array<float, kSubbands>>> correction_factors_;
  std::vector<std::array<int, kSubbands>> num_updates_;
  std::vector<std::array<size_t, kFftLengthBy2Plus1>> n_active_sections_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_SIGNAL_DEPENDENT_ERLE_ESTIMATOR_H_
