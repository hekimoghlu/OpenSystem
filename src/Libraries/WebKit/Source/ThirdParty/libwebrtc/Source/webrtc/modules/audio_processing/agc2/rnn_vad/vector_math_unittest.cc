/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 17, 2024.
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
#include "modules/audio_processing/agc2/rnn_vad/vector_math.h"

#include <vector>

#include "modules/audio_processing/agc2/cpu_features.h"
#include "test/gtest.h"

namespace webrtc {
namespace rnn_vad {
namespace {

constexpr int kSizeOfX = 19;
constexpr float kX[kSizeOfX] = {
    0.31593041f, 0.9350786f,   -0.25252445f, -0.86956251f, -0.9673632f,
    0.54571901f, -0.72504495f, -0.79509912f, -0.25525012f, -0.73340473f,
    0.15747377f, -0.04370565f, 0.76135145f,  -0.57239645f, 0.68616848f,
    0.3740298f,  0.34710799f,  -0.92207423f, 0.10738454f};
constexpr int kSizeOfXSubSpan = 16;
static_assert(kSizeOfXSubSpan < kSizeOfX, "");
constexpr float kEnergyOfX = 7.315563958160327f;
constexpr float kEnergyOfXSubspan = 6.333327669592963f;

class VectorMathParametrization
    : public ::testing::TestWithParam<AvailableCpuFeatures> {};

TEST_P(VectorMathParametrization, TestDotProduct) {
  VectorMath vector_math(/*cpu_features=*/GetParam());
  EXPECT_FLOAT_EQ(vector_math.DotProduct(kX, kX), kEnergyOfX);
  EXPECT_FLOAT_EQ(
      vector_math.DotProduct({kX, kSizeOfXSubSpan}, {kX, kSizeOfXSubSpan}),
      kEnergyOfXSubspan);
}

// Finds the relevant CPU features combinations to test.
std::vector<AvailableCpuFeatures> GetCpuFeaturesToTest() {
  std::vector<AvailableCpuFeatures> v;
  v.push_back({/*sse2=*/false, /*avx2=*/false, /*neon=*/false});
  AvailableCpuFeatures available = GetAvailableCpuFeatures();
  if (available.avx2) {
    v.push_back({/*sse2=*/false, /*avx2=*/true, /*neon=*/false});
  }
  if (available.sse2) {
    v.push_back({/*sse2=*/true, /*avx2=*/false, /*neon=*/false});
  }
  if (available.neon) {
    v.push_back({/*sse2=*/false, /*avx2=*/false, /*neon=*/true});
  }
  return v;
}

INSTANTIATE_TEST_SUITE_P(
    RnnVadTest,
    VectorMathParametrization,
    ::testing::ValuesIn(GetCpuFeaturesToTest()),
    [](const ::testing::TestParamInfo<AvailableCpuFeatures>& info) {
      return info.param.ToString();
    });

}  // namespace
}  // namespace rnn_vad
}  // namespace webrtc
