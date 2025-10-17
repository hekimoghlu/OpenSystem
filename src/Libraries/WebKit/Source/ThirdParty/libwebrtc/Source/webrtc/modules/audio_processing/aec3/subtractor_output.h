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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_SUBTRACTOR_OUTPUT_H_
#define MODULES_AUDIO_PROCESSING_AEC3_SUBTRACTOR_OUTPUT_H_

#include <array>

#include "api/array_view.h"
#include "modules/audio_processing/aec3/aec3_common.h"
#include "modules/audio_processing/aec3/fft_data.h"

namespace webrtc {

// Stores the values being returned from the echo subtractor for a single
// capture channel.
struct SubtractorOutput {
  SubtractorOutput();
  ~SubtractorOutput();

  std::array<float, kBlockSize> s_refined;
  std::array<float, kBlockSize> s_coarse;
  std::array<float, kBlockSize> e_refined;
  std::array<float, kBlockSize> e_coarse;
  FftData E_refined;
  std::array<float, kFftLengthBy2Plus1> E2_refined;
  std::array<float, kFftLengthBy2Plus1> E2_coarse;
  float s2_refined = 0.f;
  float s2_coarse = 0.f;
  float e2_refined = 0.f;
  float e2_coarse = 0.f;
  float y2 = 0.f;
  float s_refined_max_abs = 0.f;
  float s_coarse_max_abs = 0.f;

  // Reset the struct content.
  void Reset();

  // Updates the powers of the signals.
  void ComputeMetrics(rtc::ArrayView<const float> y);
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_SUBTRACTOR_OUTPUT_H_
