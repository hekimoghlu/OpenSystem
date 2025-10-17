/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 20, 2025.
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

#include "api/array_view.h"
#include "modules/audio_processing/aec3/fft_data.h"

namespace webrtc {

// Computes the power spectrum of the data.
void FftData::SpectrumAVX2(rtc::ArrayView<float> power_spectrum) const {
  RTC_DCHECK_EQ(kFftLengthBy2Plus1, power_spectrum.size());
  for (size_t k = 0; k < kFftLengthBy2; k += 8) {
    __m256 r = _mm256_loadu_ps(&re[k]);
    __m256 i = _mm256_loadu_ps(&im[k]);
    __m256 ii = _mm256_mul_ps(i, i);
    ii = _mm256_fmadd_ps(r, r, ii);
    _mm256_storeu_ps(&power_spectrum[k], ii);
  }
  power_spectrum[kFftLengthBy2] = re[kFftLengthBy2] * re[kFftLengthBy2] +
                                  im[kFftLengthBy2] * im[kFftLengthBy2];
}

}  // namespace webrtc
