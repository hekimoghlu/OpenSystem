/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 1, 2024.
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
#include "modules/audio_processing/ns/quantile_noise_estimator.h"

#include <algorithm>

#include "modules/audio_processing/ns/fast_math.h"

namespace webrtc {

QuantileNoiseEstimator::QuantileNoiseEstimator() {
  quantile_.fill(0.f);
  density_.fill(0.3f);
  log_quantile_.fill(8.f);

  constexpr float kOneBySimult = 1.f / kSimult;
  for (size_t i = 0; i < kSimult; ++i) {
    counter_[i] = floor(kLongStartupPhaseBlocks * (i + 1.f) * kOneBySimult);
  }
}

void QuantileNoiseEstimator::Estimate(
    rtc::ArrayView<const float, kFftSizeBy2Plus1> signal_spectrum,
    rtc::ArrayView<float, kFftSizeBy2Plus1> noise_spectrum) {
  std::array<float, kFftSizeBy2Plus1> log_spectrum;
  LogApproximation(signal_spectrum, log_spectrum);

  int quantile_index_to_return = -1;
  // Loop over simultaneous estimates.
  for (int s = 0, k = 0; s < kSimult;
       ++s, k += static_cast<int>(kFftSizeBy2Plus1)) {
    const float one_by_counter_plus_1 = 1.f / (counter_[s] + 1.f);
    for (int i = 0, j = k; i < static_cast<int>(kFftSizeBy2Plus1); ++i, ++j) {
      // Update log quantile estimate.
      const float delta = density_[j] > 1.f ? 40.f / density_[j] : 40.f;

      const float multiplier = delta * one_by_counter_plus_1;
      if (log_spectrum[i] > log_quantile_[j]) {
        log_quantile_[j] += 0.25f * multiplier;
      } else {
        log_quantile_[j] -= 0.75f * multiplier;
      }

      // Update density estimate.
      constexpr float kWidth = 0.01f;
      constexpr float kOneByWidthPlus2 = 1.f / (2.f * kWidth);
      if (fabs(log_spectrum[i] - log_quantile_[j]) < kWidth) {
        density_[j] = (counter_[s] * density_[j] + kOneByWidthPlus2) *
                      one_by_counter_plus_1;
      }
    }

    if (counter_[s] >= kLongStartupPhaseBlocks) {
      counter_[s] = 0;
      if (num_updates_ >= kLongStartupPhaseBlocks) {
        quantile_index_to_return = k;
      }
    }

    ++counter_[s];
  }

  // Sequentially update the noise during startup.
  if (num_updates_ < kLongStartupPhaseBlocks) {
    // Use the last "s" to get noise during startup that differ from zero.
    quantile_index_to_return = kFftSizeBy2Plus1 * (kSimult - 1);
    ++num_updates_;
  }

  if (quantile_index_to_return >= 0) {
    ExpApproximation(
        rtc::ArrayView<const float>(&log_quantile_[quantile_index_to_return],
                                    kFftSizeBy2Plus1),
        quantile_);
  }

  std::copy(quantile_.begin(), quantile_.end(), noise_spectrum.begin());
}

}  // namespace webrtc
