/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 24, 2022.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_SUPPRESSION_FILTER_H_
#define MODULES_AUDIO_PROCESSING_AEC3_SUPPRESSION_FILTER_H_

#include <array>
#include <vector>

#include "modules/audio_processing/aec3/aec3_common.h"
#include "modules/audio_processing/aec3/aec3_fft.h"
#include "modules/audio_processing/aec3/block.h"
#include "modules/audio_processing/aec3/fft_data.h"

namespace webrtc {

class SuppressionFilter {
 public:
  SuppressionFilter(Aec3Optimization optimization,
                    int sample_rate_hz,
                    size_t num_capture_channels_);
  ~SuppressionFilter();

  SuppressionFilter(const SuppressionFilter&) = delete;
  SuppressionFilter& operator=(const SuppressionFilter&) = delete;

  void ApplyGain(rtc::ArrayView<const FftData> comfort_noise,
                 rtc::ArrayView<const FftData> comfort_noise_high_bands,
                 const std::array<float, kFftLengthBy2Plus1>& suppression_gain,
                 float high_bands_gain,
                 rtc::ArrayView<const FftData> E_lowest_band,
                 Block* e);

 private:
  const Aec3Optimization optimization_;
  const int sample_rate_hz_;
  const size_t num_capture_channels_;
  const Aec3Fft fft_;
  std::vector<std::vector<std::array<float, kFftLengthBy2>>> e_output_old_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_SUPPRESSION_FILTER_H_
