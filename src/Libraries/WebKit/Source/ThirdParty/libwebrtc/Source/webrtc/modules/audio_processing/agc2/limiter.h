/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 22, 2021.
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
#ifndef MODULES_AUDIO_PROCESSING_AGC2_LIMITER_H_
#define MODULES_AUDIO_PROCESSING_AGC2_LIMITER_H_

#include <vector>

#include "absl/strings/string_view.h"
#include "api/audio/audio_frame.h"
#include "modules/audio_processing/agc2/fixed_digital_level_estimator.h"
#include "modules/audio_processing/agc2/interpolated_gain_curve.h"
#include "modules/audio_processing/include/audio_frame_view.h"

namespace webrtc {
class ApmDataDumper;

class Limiter {
 public:
  // See `SetSamplesPerChannel()` for valid values for `samples_per_channel`.
  Limiter(ApmDataDumper* apm_data_dumper,
          size_t samples_per_channel,
          absl::string_view histogram_name_prefix);

  Limiter(const Limiter& limiter) = delete;
  Limiter& operator=(const Limiter& limiter) = delete;
  ~Limiter();

  // Applies limiter and hard-clipping to `signal`.
  void Process(DeinterleavedView<float> signal);

  InterpolatedGainCurve::Stats GetGainCurveStats() const;

  // Supported values must be
  // * Supported by FixedDigitalLevelEstimator
  // * Below or equal to kMaximalNumberOfSamplesPerChannel so that samples
  //   fit in the per_sample_scaling_factors_ array.
  void SetSamplesPerChannel(size_t samples_per_channel);

  // Resets the internal state.
  void Reset();

  float LastAudioLevel() const;

 private:
  const InterpolatedGainCurve interp_gain_curve_;
  FixedDigitalLevelEstimator level_estimator_;
  ApmDataDumper* const apm_data_dumper_ = nullptr;

  // Work array containing the sub-frame scaling factors to be interpolated.
  std::array<float, kSubFramesInFrame + 1> scaling_factors_ = {};
  std::array<float, kMaximalNumberOfSamplesPerChannel>
      per_sample_scaling_factors_ = {};
  float last_scaling_factor_ = 1.f;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AGC2_LIMITER_H_
