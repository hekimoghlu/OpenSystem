/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 13, 2021.
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
// MSVC++ requires this to be set before any other includes to get M_PI.
#define _USE_MATH_DEFINES
#include "modules/audio_processing/rms_level.h"

#include <cmath>
#include <memory>
#include <vector>

#include "api/array_view.h"
#include "rtc_base/checks.h"
#include "rtc_base/numerics/safe_conversions.h"
#include "test/gtest.h"

namespace webrtc {
namespace {
constexpr int kSampleRateHz = 48000;
constexpr size_t kBlockSizeSamples = kSampleRateHz / 100;

std::unique_ptr<RmsLevel> RunTest(rtc::ArrayView<const int16_t> input) {
  std::unique_ptr<RmsLevel> level(new RmsLevel);
  for (size_t n = 0; n + kBlockSizeSamples <= input.size();
       n += kBlockSizeSamples) {
    level->Analyze(input.subview(n, kBlockSizeSamples));
  }
  return level;
}

std::unique_ptr<RmsLevel> RunTest(rtc::ArrayView<const float> input) {
  std::unique_ptr<RmsLevel> level(new RmsLevel);
  for (size_t n = 0; n + kBlockSizeSamples <= input.size();
       n += kBlockSizeSamples) {
    level->Analyze(input.subview(n, kBlockSizeSamples));
  }
  return level;
}

std::vector<int16_t> CreateInt16Sinusoid(int frequency_hz,
                                         int amplitude,
                                         size_t num_samples) {
  std::vector<int16_t> x(num_samples);
  for (size_t n = 0; n < num_samples; ++n) {
    x[n] = rtc::saturated_cast<int16_t>(
        amplitude * std::sin(2 * M_PI * n * frequency_hz / kSampleRateHz));
  }
  return x;
}

std::vector<float> CreateFloatSinusoid(int frequency_hz,
                                       int amplitude,
                                       size_t num_samples) {
  std::vector<int16_t> x16 =
      CreateInt16Sinusoid(frequency_hz, amplitude, num_samples);
  std::vector<float> x(x16.size());
  for (size_t n = 0; n < x.size(); ++n) {
    x[n] = x16[n];
  }
  return x;
}

}  // namespace

TEST(RmsLevelTest, VerifyIndentityBetweenFloatAndFix) {
  auto x_f = CreateFloatSinusoid(1000, INT16_MAX, kSampleRateHz);
  auto x_i = CreateFloatSinusoid(1000, INT16_MAX, kSampleRateHz);
  auto level_f = RunTest(x_f);
  auto level_i = RunTest(x_i);
  int avg_i = level_i->Average();
  int avg_f = level_f->Average();
  EXPECT_EQ(3, avg_i);  // -3 dBFS
  EXPECT_EQ(avg_f, avg_i);
}

TEST(RmsLevelTest, Run1000HzFullScale) {
  auto x = CreateInt16Sinusoid(1000, INT16_MAX, kSampleRateHz);
  auto level = RunTest(x);
  EXPECT_EQ(3, level->Average());  // -3 dBFS
}

TEST(RmsLevelTest, Run1000HzFullScaleAverageAndPeak) {
  auto x = CreateInt16Sinusoid(1000, INT16_MAX, kSampleRateHz);
  auto level = RunTest(x);
  auto stats = level->AverageAndPeak();
  EXPECT_EQ(3, stats.average);  // -3 dBFS
  EXPECT_EQ(3, stats.peak);
}

TEST(RmsLevelTest, Run1000HzHalfScale) {
  auto x = CreateInt16Sinusoid(1000, INT16_MAX / 2, kSampleRateHz);
  auto level = RunTest(x);
  EXPECT_EQ(9, level->Average());  // -9 dBFS
}

TEST(RmsLevelTest, RunZeros) {
  std::vector<int16_t> x(kSampleRateHz, 0);  // 1 second of pure silence.
  auto level = RunTest(x);
  EXPECT_EQ(127, level->Average());
}

TEST(RmsLevelTest, RunZerosAverageAndPeak) {
  std::vector<int16_t> x(kSampleRateHz, 0);  // 1 second of pure silence.
  auto level = RunTest(x);
  auto stats = level->AverageAndPeak();
  EXPECT_EQ(127, stats.average);
  EXPECT_EQ(127, stats.peak);
}

TEST(RmsLevelTest, NoSamples) {
  RmsLevel level;
  EXPECT_EQ(127, level.Average());  // Return minimum if no samples are given.
}

TEST(RmsLevelTest, NoSamplesAverageAndPeak) {
  RmsLevel level;
  auto stats = level.AverageAndPeak();
  EXPECT_EQ(127, stats.average);
  EXPECT_EQ(127, stats.peak);
}

TEST(RmsLevelTest, PollTwice) {
  auto x = CreateInt16Sinusoid(1000, INT16_MAX, kSampleRateHz);
  auto level = RunTest(x);
  level->Average();
  EXPECT_EQ(127, level->Average());  // Stats should be reset at this point.
}

TEST(RmsLevelTest, Reset) {
  auto x = CreateInt16Sinusoid(1000, INT16_MAX, kSampleRateHz);
  auto level = RunTest(x);
  level->Reset();
  EXPECT_EQ(127, level->Average());  // Stats should be reset at this point.
}

// Inserts 1 second of full-scale sinusoid, followed by 1 second of muted.
TEST(RmsLevelTest, ProcessMuted) {
  auto x = CreateInt16Sinusoid(1000, INT16_MAX, kSampleRateHz);
  auto level = RunTest(x);
  const size_t kBlocksPerSecond = rtc::CheckedDivExact(
      static_cast<size_t>(kSampleRateHz), kBlockSizeSamples);
  for (size_t i = 0; i < kBlocksPerSecond; ++i) {
    level->AnalyzeMuted(kBlockSizeSamples);
  }
  EXPECT_EQ(6, level->Average());  // Average RMS halved due to the silence.
}

// Digital silence must yield 127 and anything else should yield 126 or lower.
TEST(RmsLevelTest, OnlyDigitalSilenceIs127) {
  std::vector<int16_t> test_buffer(kSampleRateHz, 0);
  auto level = RunTest(test_buffer);
  EXPECT_EQ(127, level->Average());
  // Change one sample to something other than 0 to make the buffer not strictly
  // represent digital silence.
  test_buffer[0] = 1;
  level = RunTest(test_buffer);
  EXPECT_LT(level->Average(), 127);
}

// Inserts 1 second of half-scale sinusoid, follwed by 10 ms of full-scale, and
// finally 1 second of half-scale again. Expect the average to be -9 dBFS due
// to the vast majority of the signal being half-scale, and the peak to be
// -3 dBFS.
TEST(RmsLevelTest, RunHalfScaleAndInsertFullScale) {
  auto half_scale = CreateInt16Sinusoid(1000, INT16_MAX / 2, kSampleRateHz);
  auto full_scale = CreateInt16Sinusoid(1000, INT16_MAX, kSampleRateHz / 100);
  auto x = half_scale;
  x.insert(x.end(), full_scale.begin(), full_scale.end());
  x.insert(x.end(), half_scale.begin(), half_scale.end());
  ASSERT_EQ(static_cast<size_t>(2 * kSampleRateHz + kSampleRateHz / 100),
            x.size());
  auto level = RunTest(x);
  auto stats = level->AverageAndPeak();
  EXPECT_EQ(9, stats.average);
  EXPECT_EQ(3, stats.peak);
}

TEST(RmsLevelTest, ResetOnBlockSizeChange) {
  auto x = CreateInt16Sinusoid(1000, INT16_MAX, kSampleRateHz);
  auto level = RunTest(x);
  // Create a new signal with half amplitude, but double block length.
  auto y = CreateInt16Sinusoid(1000, INT16_MAX / 2, kBlockSizeSamples * 2);
  level->Analyze(y);
  auto stats = level->AverageAndPeak();
  // Expect all stats to only be influenced by the last signal (y), since the
  // changed block size should reset the stats.
  EXPECT_EQ(9, stats.average);
  EXPECT_EQ(9, stats.peak);
}

}  // namespace webrtc
