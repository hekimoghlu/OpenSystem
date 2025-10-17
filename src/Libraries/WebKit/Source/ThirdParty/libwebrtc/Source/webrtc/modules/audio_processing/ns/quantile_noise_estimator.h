/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 20, 2022.
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
#ifndef MODULES_AUDIO_PROCESSING_NS_QUANTILE_NOISE_ESTIMATOR_H_
#define MODULES_AUDIO_PROCESSING_NS_QUANTILE_NOISE_ESTIMATOR_H_

#include <math.h>

#include <array>

#include "api/array_view.h"
#include "modules/audio_processing/ns/ns_common.h"

namespace webrtc {

constexpr int kSimult = 3;

// For quantile noise estimation.
class QuantileNoiseEstimator {
 public:
  QuantileNoiseEstimator();
  QuantileNoiseEstimator(const QuantileNoiseEstimator&) = delete;
  QuantileNoiseEstimator& operator=(const QuantileNoiseEstimator&) = delete;

  // Estimate noise.
  void Estimate(rtc::ArrayView<const float, kFftSizeBy2Plus1> signal_spectrum,
                rtc::ArrayView<float, kFftSizeBy2Plus1> noise_spectrum);

 private:
  std::array<float, kSimult * kFftSizeBy2Plus1> density_;
  std::array<float, kSimult * kFftSizeBy2Plus1> log_quantile_;
  std::array<float, kFftSizeBy2Plus1> quantile_;
  std::array<int, kSimult> counter_;
  int num_updates_ = 1;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_NS_QUANTILE_NOISE_ESTIMATOR_H_
