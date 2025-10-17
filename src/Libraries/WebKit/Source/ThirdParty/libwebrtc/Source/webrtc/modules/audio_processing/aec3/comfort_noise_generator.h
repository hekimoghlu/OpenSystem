/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 30, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_COMFORT_NOISE_GENERATOR_H_
#define MODULES_AUDIO_PROCESSING_AEC3_COMFORT_NOISE_GENERATOR_H_

#include <stdint.h>

#include <array>
#include <memory>

#include "modules/audio_processing/aec3/aec3_common.h"
#include "modules/audio_processing/aec3/aec_state.h"
#include "modules/audio_processing/aec3/fft_data.h"
#include "rtc_base/system/arch.h"

namespace webrtc {
namespace aec3 {
#if defined(WEBRTC_ARCH_X86_FAMILY)

void EstimateComfortNoise_SSE2(const std::array<float, kFftLengthBy2Plus1>& N2,
                               uint32_t* seed,
                               FftData* lower_band_noise,
                               FftData* upper_band_noise);
#endif
void EstimateComfortNoise(const std::array<float, kFftLengthBy2Plus1>& N2,
                          uint32_t* seed,
                          FftData* lower_band_noise,
                          FftData* upper_band_noise);

}  // namespace aec3

// Generates the comfort noise.
class ComfortNoiseGenerator {
 public:
  ComfortNoiseGenerator(const EchoCanceller3Config& config,
                        Aec3Optimization optimization,
                        size_t num_capture_channels);
  ComfortNoiseGenerator() = delete;
  ~ComfortNoiseGenerator();
  ComfortNoiseGenerator(const ComfortNoiseGenerator&) = delete;

  // Computes the comfort noise.
  void Compute(bool saturated_capture,
               rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>>
                   capture_spectrum,
               rtc::ArrayView<FftData> lower_band_noise,
               rtc::ArrayView<FftData> upper_band_noise);

  // Returns the estimate of the background noise spectrum.
  rtc::ArrayView<const std::array<float, kFftLengthBy2Plus1>> NoiseSpectrum()
      const {
    return N2_;
  }

 private:
  const Aec3Optimization optimization_;
  uint32_t seed_;
  const size_t num_capture_channels_;
  const float noise_floor_;
  std::unique_ptr<std::vector<std::array<float, kFftLengthBy2Plus1>>>
      N2_initial_;
  std::vector<std::array<float, kFftLengthBy2Plus1>> Y2_smoothed_;
  std::vector<std::array<float, kFftLengthBy2Plus1>> N2_;
  int N2_counter_ = 0;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_COMFORT_NOISE_GENERATOR_H_
