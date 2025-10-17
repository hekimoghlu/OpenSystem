/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 17, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_ERL_ESTIMATOR_H_
#define MODULES_AUDIO_PROCESSING_AEC3_ERL_ESTIMATOR_H_

#include <stddef.h>

#include <array>
#include <vector>

#include "api/array_view.h"
#include "modules/audio_processing/aec3/aec3_common.h"

namespace webrtc {

// Estimates the echo return loss based on the signal spectra.
class ErlEstimator {
 public:
  explicit ErlEstimator(size_t startup_phase_length_blocks_);
  ~ErlEstimator();

  ErlEstimator(const ErlEstimator&) = delete;
  ErlEstimator& operator=(const ErlEstimator&) = delete;

  // Resets the ERL estimation.
  void Reset();

  // Updates the ERL estimate.
  void Update(const std::vector<bool>& converged_filters,
              rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>>
                  render_spectra,
              rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>>
                  capture_spectra);

  // Returns the most recent ERL estimate.
  const std::array<float, kFftLengthBy2Plus1>& Erl() const { return erl_; }
  float ErlTimeDomain() const { return erl_time_domain_; }

 private:
  const size_t startup_phase_length_blocks__;
  std::array<float, kFftLengthBy2Plus1> erl_;
  std::array<int, kFftLengthBy2Minus1> hold_counters_;
  float erl_time_domain_;
  int hold_counter_time_domain_;
  size_t blocks_since_reset_ = 0;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_ERL_ESTIMATOR_H_
