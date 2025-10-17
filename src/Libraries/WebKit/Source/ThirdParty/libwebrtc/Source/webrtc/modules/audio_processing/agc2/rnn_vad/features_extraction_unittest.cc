/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 15, 2024.
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
#include "modules/audio_processing/agc2/rnn_vad/features_extraction.h"

#include <cmath>
#include <vector>

#include "modules/audio_processing/agc2/cpu_features.h"
#include "rtc_base/numerics/safe_compare.h"
#include "rtc_base/numerics/safe_conversions.h"
// TODO(bugs.webrtc.org/8948): Add when the issue is fixed.
// #include "test/fpe_observer.h"
#include "test/gtest.h"

namespace webrtc {
namespace rnn_vad {
namespace {

constexpr int ceil(int n, int m) {
  return (n + m - 1) / m;
}

// Number of 10 ms frames required to fill a pitch buffer having size
// `kBufSize24kHz`.
constexpr int kNumTestDataFrames = ceil(kBufSize24kHz, kFrameSize10ms24kHz);
// Number of samples for the test data.
constexpr int kNumTestDataSize = kNumTestDataFrames * kFrameSize10ms24kHz;

// Verifies that the pitch in Hz is in the detectable range.
bool PitchIsValid(float pitch_hz) {
  const int pitch_period = static_cast<float>(kSampleRate24kHz) / pitch_hz;
  return kInitialMinPitch24kHz <= pitch_period &&
         pitch_period <= kMaxPitch24kHz;
}

void CreatePureTone(float amplitude, float freq_hz, rtc::ArrayView<float> dst) {
  for (int i = 0; rtc::SafeLt(i, dst.size()); ++i) {
    dst[i] = amplitude * std::sin(2.f * kPi * freq_hz * i / kSampleRate24kHz);
  }
}

// Feeds `features_extractor` with `samples` splitting it in 10 ms frames.
// For every frame, the output is written into `feature_vector`. Returns true
// if silence is detected in the last frame.
bool FeedTestData(FeaturesExtractor& features_extractor,
                  rtc::ArrayView<const float> samples,
                  rtc::ArrayView<float, kFeatureVectorSize> feature_vector) {
  // TODO(bugs.webrtc.org/8948): Add when the issue is fixed.
  // FloatingPointExceptionObserver fpe_observer;
  bool is_silence = true;
  const int num_frames = samples.size() / kFrameSize10ms24kHz;
  for (int i = 0; i < num_frames; ++i) {
    is_silence = features_extractor.CheckSilenceComputeFeatures(
        {samples.data() + i * kFrameSize10ms24kHz, kFrameSize10ms24kHz},
        feature_vector);
  }
  return is_silence;
}

// Extracts the features for two pure tones and verifies that the pitch field
// values reflect the known tone frequencies.
TEST(RnnVadTest, FeatureExtractionLowHighPitch) {
  constexpr float amplitude = 1000.f;
  constexpr float low_pitch_hz = 150.f;
  constexpr float high_pitch_hz = 250.f;
  ASSERT_TRUE(PitchIsValid(low_pitch_hz));
  ASSERT_TRUE(PitchIsValid(high_pitch_hz));

  const AvailableCpuFeatures cpu_features = GetAvailableCpuFeatures();
  FeaturesExtractor features_extractor(cpu_features);
  std::vector<float> samples(kNumTestDataSize);
  std::vector<float> feature_vector(kFeatureVectorSize);
  ASSERT_EQ(kFeatureVectorSize, rtc::dchecked_cast<int>(feature_vector.size()));
  rtc::ArrayView<float, kFeatureVectorSize> feature_vector_view(
      feature_vector.data(), kFeatureVectorSize);

  // Extract the normalized scalar feature that is proportional to the estimated
  // pitch period.
  constexpr int pitch_feature_index = kFeatureVectorSize - 2;
  // Low frequency tone - i.e., high period.
  CreatePureTone(amplitude, low_pitch_hz, samples);
  ASSERT_FALSE(FeedTestData(features_extractor, samples, feature_vector_view));
  float high_pitch_period = feature_vector_view[pitch_feature_index];
  // High frequency tone - i.e., low period.
  features_extractor.Reset();
  CreatePureTone(amplitude, high_pitch_hz, samples);
  ASSERT_FALSE(FeedTestData(features_extractor, samples, feature_vector_view));
  float low_pitch_period = feature_vector_view[pitch_feature_index];
  // Check.
  EXPECT_LT(low_pitch_period, high_pitch_period);
}

}  // namespace
}  // namespace rnn_vad
}  // namespace webrtc
