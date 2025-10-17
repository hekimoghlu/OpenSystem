/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 3, 2023.
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
#include "modules/audio_processing/ns/histograms.h"

namespace webrtc {

Histograms::Histograms() {
  Clear();
}

void Histograms::Clear() {
  lrt_.fill(0);
  spectral_flatness_.fill(0);
  spectral_diff_.fill(0);
}

void Histograms::Update(const SignalModel& features_) {
  // Update the histogram for the LRT.
  constexpr float kOneByBinSizeLrt = 1.f / kBinSizeLrt;
  if (features_.lrt < kHistogramSize * kBinSizeLrt && features_.lrt >= 0.f) {
    ++lrt_[kOneByBinSizeLrt * features_.lrt];
  }

  // Update histogram for the spectral flatness.
  constexpr float kOneByBinSizeSpecFlat = 1.f / kBinSizeSpecFlat;
  if (features_.spectral_flatness < kHistogramSize * kBinSizeSpecFlat &&
      features_.spectral_flatness >= 0.f) {
    ++spectral_flatness_[features_.spectral_flatness * kOneByBinSizeSpecFlat];
  }

  // Update histogram for the spectral difference.
  constexpr float kOneByBinSizeSpecDiff = 1.f / kBinSizeSpecDiff;
  if (features_.spectral_diff < kHistogramSize * kBinSizeSpecDiff &&
      features_.spectral_diff >= 0.f) {
    ++spectral_diff_[features_.spectral_diff * kOneByBinSizeSpecDiff];
  }
}

}  // namespace webrtc
