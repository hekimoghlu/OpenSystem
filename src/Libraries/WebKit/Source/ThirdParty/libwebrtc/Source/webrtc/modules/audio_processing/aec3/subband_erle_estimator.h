/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 1, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_SUBBAND_ERLE_ESTIMATOR_H_
#define MODULES_AUDIO_PROCESSING_AEC3_SUBBAND_ERLE_ESTIMATOR_H_

#include <stddef.h>

#include <array>
#include <memory>
#include <vector>

#include "api/array_view.h"
#include "api/audio/echo_canceller3_config.h"
#include "modules/audio_processing/aec3/aec3_common.h"
#include "modules/audio_processing/logging/apm_data_dumper.h"

namespace webrtc {

// Estimates the echo return loss enhancement for each frequency subband.
class SubbandErleEstimator {
 public:
  SubbandErleEstimator(const EchoCanceller3Config& config,
                       size_t num_capture_channels);
  ~SubbandErleEstimator();

  // Resets the ERLE estimator.
  void Reset();

  // Updates the ERLE estimate.
  void Update(rtc::ArrayView<const float, kFftLengthBy2Plus1> X2,
              rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>> Y2,
              rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>> E2,
              const std::vector<bool>& converged_filters);

  // Returns the ERLE estimate.
  rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>> Erle(
      bool onset_compensated) const {
    return onset_compensated && use_onset_detection_ ? erle_onset_compensated_
                                                     : erle_;
  }

  // Returns the non-capped ERLE estimate.
  rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>> ErleUnbounded()
      const {
    return erle_unbounded_;
  }

  // Returns the ERLE estimate at onsets (only used for testing).
  rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>> ErleDuringOnsets()
      const {
    return erle_during_onsets_;
  }

  void Dump(const std::unique_ptr<ApmDataDumper>& data_dumper) const;

 private:
  struct AccumulatedSpectra {
    explicit AccumulatedSpectra(size_t num_capture_channels)
        : Y2(num_capture_channels),
          E2(num_capture_channels),
          low_render_energy(num_capture_channels),
          num_points(num_capture_channels) {}
    std::vector<std::array<float, kFftLengthBy2Plus1>> Y2;
    std::vector<std::array<float, kFftLengthBy2Plus1>> E2;
    std::vector<std::array<bool, kFftLengthBy2Plus1>> low_render_energy;
    std::vector<int> num_points;
  };

  void UpdateAccumulatedSpectra(
      rtc::ArrayView<const float, kFftLengthBy2Plus1> X2,
      rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>> Y2,
      rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>> E2,
      const std::vector<bool>& converged_filters);

  void ResetAccumulatedSpectra();

  void UpdateBands(const std::vector<bool>& converged_filters);
  void DecreaseErlePerBandForLowRenderSignals();

  const bool use_onset_detection_;
  const float min_erle_;
  const std::array<float, kFftLengthBy2Plus1> max_erle_;
  const bool use_min_erle_during_onsets_;
  AccumulatedSpectra accum_spectra_;
  // ERLE without special handling of render onsets.
  std::vector<std::array<float, kFftLengthBy2Plus1>> erle_;
  // ERLE lowered during render onsets.
  std::vector<std::array<float, kFftLengthBy2Plus1>> erle_onset_compensated_;
  std::vector<std::array<float, kFftLengthBy2Plus1>> erle_unbounded_;
  // Estimation of ERLE during render onsets.
  std::vector<std::array<float, kFftLengthBy2Plus1>> erle_during_onsets_;
  std::vector<std::array<bool, kFftLengthBy2Plus1>> coming_onset_;
  std::vector<std::array<int, kFftLengthBy2Plus1>> hold_counters_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_SUBBAND_ERLE_ESTIMATOR_H_
