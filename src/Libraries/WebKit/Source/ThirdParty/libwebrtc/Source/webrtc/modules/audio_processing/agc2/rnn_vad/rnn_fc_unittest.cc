/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 15, 2023.
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
#include "modules/audio_processing/agc2/rnn_vad/rnn_fc.h"

#include <array>
#include <vector>

#include "api/array_view.h"
#include "modules/audio_processing/agc2/cpu_features.h"
#include "modules/audio_processing/agc2/rnn_vad/test_utils.h"
#include "modules/audio_processing/test/performance_timer.h"
#include "rtc_base/logging.h"
#include "rtc_base/system/arch.h"
#include "test/gtest.h"
#include "third_party/rnnoise/src/rnn_vad_weights.h"

namespace webrtc {
namespace rnn_vad {
namespace {

using ::rnnoise::kInputDenseBias;
using ::rnnoise::kInputDenseWeights;
using ::rnnoise::kInputLayerInputSize;
using ::rnnoise::kInputLayerOutputSize;

// Fully connected layer test data.
constexpr std::array<float, 42> kFullyConnectedInputVector = {
    -1.00131f,   -0.627069f, -7.81097f,  7.86285f,    -2.87145f,  3.32365f,
    -0.653161f,  0.529839f,  -0.425307f, 0.25583f,    0.235094f,  0.230527f,
    -0.144687f,  0.182785f,  0.57102f,   0.125039f,   0.479482f,  -0.0255439f,
    -0.0073141f, -0.147346f, -0.217106f, -0.0846906f, -8.34943f,  3.09065f,
    1.42628f,    -0.85235f,  -0.220207f, -0.811163f,  2.09032f,   -2.01425f,
    -0.690268f,  -0.925327f, -0.541354f, 0.58455f,    -0.606726f, -0.0372358f,
    0.565991f,   0.435854f,  0.420812f,  0.162198f,   -2.13f,     10.0089f};
constexpr std::array<float, 24> kFullyConnectedExpectedOutput = {
    -0.623293f, -0.988299f, 0.999378f,  0.967168f,  0.103087f,  -0.978545f,
    -0.856347f, 0.346675f,  1.f,        -0.717442f, -0.544176f, 0.960363f,
    0.983443f,  0.999991f,  -0.824335f, 0.984742f,  0.990208f,  0.938179f,
    0.875092f,  0.999846f,  0.997707f,  -0.999382f, 0.973153f,  -0.966605f};

class RnnFcParametrization
    : public ::testing::TestWithParam<AvailableCpuFeatures> {};

// Checks that the output of a fully connected layer is within tolerance given
// test input data.
TEST_P(RnnFcParametrization, CheckFullyConnectedLayerOutput) {
  FullyConnectedLayer fc(kInputLayerInputSize, kInputLayerOutputSize,
                         kInputDenseBias, kInputDenseWeights,
                         ActivationFunction::kTansigApproximated,
                         /*cpu_features=*/GetParam(),
                         /*layer_name=*/"FC");
  fc.ComputeOutput(kFullyConnectedInputVector);
  ExpectNearAbsolute(kFullyConnectedExpectedOutput, fc, 1e-5f);
}

TEST_P(RnnFcParametrization, DISABLED_BenchmarkFullyConnectedLayer) {
  const AvailableCpuFeatures cpu_features = GetParam();
  FullyConnectedLayer fc(kInputLayerInputSize, kInputLayerOutputSize,
                         kInputDenseBias, kInputDenseWeights,
                         ActivationFunction::kTansigApproximated, cpu_features,
                         /*layer_name=*/"FC");

  constexpr int kNumTests = 10000;
  ::webrtc::test::PerformanceTimer perf_timer(kNumTests);
  for (int k = 0; k < kNumTests; ++k) {
    perf_timer.StartTimer();
    fc.ComputeOutput(kFullyConnectedInputVector);
    perf_timer.StopTimer();
  }
  RTC_LOG(LS_INFO) << "CPU features: " << cpu_features.ToString() << " | "
                   << (perf_timer.GetDurationAverage() / 1000) << " +/- "
                   << (perf_timer.GetDurationStandardDeviation() / 1000)
                   << " ms";
}

// Finds the relevant CPU features combinations to test.
std::vector<AvailableCpuFeatures> GetCpuFeaturesToTest() {
  std::vector<AvailableCpuFeatures> v;
  v.push_back(NoAvailableCpuFeatures());
  AvailableCpuFeatures available = GetAvailableCpuFeatures();
  if (available.sse2) {
    v.push_back({/*sse2=*/true, /*avx2=*/false, /*neon=*/false});
  }
  if (available.avx2) {
    v.push_back({/*sse2=*/false, /*avx2=*/true, /*neon=*/false});
  }
  if (available.neon) {
    v.push_back({/*sse2=*/false, /*avx2=*/false, /*neon=*/true});
  }
  return v;
}

INSTANTIATE_TEST_SUITE_P(
    RnnVadTest,
    RnnFcParametrization,
    ::testing::ValuesIn(GetCpuFeaturesToTest()),
    [](const ::testing::TestParamInfo<AvailableCpuFeatures>& info) {
      return info.param.ToString();
    });

}  // namespace
}  // namespace rnn_vad
}  // namespace webrtc
