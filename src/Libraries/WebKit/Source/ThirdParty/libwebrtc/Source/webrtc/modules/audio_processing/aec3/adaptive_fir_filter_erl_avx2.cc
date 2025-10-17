/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 7, 2024.
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
#include <immintrin.h>

#include "modules/audio_processing/aec3/adaptive_fir_filter_erl.h"

namespace webrtc {

namespace aec3 {

// Computes and stores the echo return loss estimate of the filter, which is the
// sum of the partition frequency responses.
void ErlComputer_AVX2(
    const std::vector<std::array<float, kFftLengthBy2Plus1>>& H2,
    rtc::ArrayView<float> erl) {
  std::fill(erl.begin(), erl.end(), 0.f);
  for (auto& H2_j : H2) {
    for (size_t k = 0; k < kFftLengthBy2; k += 8) {
      const __m256 H2_j_k = _mm256_loadu_ps(&H2_j[k]);
      __m256 erl_k = _mm256_loadu_ps(&erl[k]);
      erl_k = _mm256_add_ps(erl_k, H2_j_k);
      _mm256_storeu_ps(&erl[k], erl_k);
    }
    erl[kFftLengthBy2] += H2_j[kFftLengthBy2];
  }
}

}  // namespace aec3
}  // namespace webrtc
