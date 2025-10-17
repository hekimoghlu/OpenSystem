/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 17, 2023.
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
#include "modules/audio_processing/aec3/comfort_noise_generator.h"

#include <algorithm>
#include <numeric>

#include "api/audio/echo_canceller3_config.h"
#include "modules/audio_processing/aec3/aec_state.h"
#include "rtc_base/random.h"
#include "rtc_base/system/arch.h"
#include "system_wrappers/include/cpu_features_wrapper.h"
#include "test/gtest.h"

namespace webrtc {
namespace aec3 {
namespace {

float Power(const FftData& N) {
  std::array<float, kFftLengthBy2Plus1> N2;
  N.Spectrum(Aec3Optimization::kNone, N2);
  return std::accumulate(N2.begin(), N2.end(), 0.f) / N2.size();
}

}  // namespace

TEST(ComfortNoiseGenerator, CorrectLevel) {
  constexpr size_t kNumChannels = 5;
  EchoCanceller3Config config;
  ComfortNoiseGenerator cng(config, DetectOptimization(), kNumChannels);
  AecState aec_state(config, kNumChannels);

  std::vector<std::array<float, kFftLengthBy2Plus1>> N2(kNumChannels);
  std::vector<FftData> n_lower(kNumChannels);
  std::vector<FftData> n_upper(kNumChannels);

  for (size_t ch = 0; ch < kNumChannels; ++ch) {
    N2[ch].fill(1000.f * 1000.f / (ch + 1));
    n_lower[ch].re.fill(0.f);
    n_lower[ch].im.fill(0.f);
    n_upper[ch].re.fill(0.f);
    n_upper[ch].im.fill(0.f);
  }

  // Ensure instantaneous updata to nonzero noise.
  cng.Compute(false, N2, n_lower, n_upper);

  for (size_t ch = 0; ch < kNumChannels; ++ch) {
    EXPECT_LT(0.f, Power(n_lower[ch]));
    EXPECT_LT(0.f, Power(n_upper[ch]));
  }

  for (int k = 0; k < 10000; ++k) {
    cng.Compute(false, N2, n_lower, n_upper);
  }

  for (size_t ch = 0; ch < kNumChannels; ++ch) {
    EXPECT_NEAR(2.f * N2[ch][0], Power(n_lower[ch]), N2[ch][0] / 10.f);
    EXPECT_NEAR(2.f * N2[ch][0], Power(n_upper[ch]), N2[ch][0] / 10.f);
  }
}

}  // namespace aec3
}  // namespace webrtc
