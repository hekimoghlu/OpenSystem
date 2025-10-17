/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 3, 2024.
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
#ifndef MODULES_AUDIO_PROCESSING_AGC2_NOISE_LEVEL_ESTIMATOR_H_
#define MODULES_AUDIO_PROCESSING_AGC2_NOISE_LEVEL_ESTIMATOR_H_

#include <memory>

#include "api/audio/audio_view.h"

namespace webrtc {
class ApmDataDumper;

// Noise level estimator interface.
class NoiseLevelEstimator {
 public:
  virtual ~NoiseLevelEstimator() = default;
  // Analyzes a 10 ms `frame`, updates the noise level estimation and returns
  // the value for the latter in dBFS.
  virtual float Analyze(DeinterleavedView<const float> frame) = 0;
};

// Creates a noise level estimator based on noise floor detection.
std::unique_ptr<NoiseLevelEstimator> CreateNoiseFloorEstimator(
    ApmDataDumper* data_dumper);

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AGC2_NOISE_LEVEL_ESTIMATOR_H_
