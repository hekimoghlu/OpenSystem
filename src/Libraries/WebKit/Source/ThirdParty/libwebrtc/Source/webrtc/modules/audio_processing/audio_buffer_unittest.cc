/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 19, 2024.
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
#include "modules/audio_processing/audio_buffer.h"

#include <cmath>

#include "api/audio/audio_view.h"
#include "test/gtest.h"
#include "test/testsupport/rtc_expect_death.h"

namespace webrtc {

namespace {

const size_t kSampleRateHz = 48000u;
const size_t kStereo = 2u;
const size_t kMono = 1u;

void ExpectNumChannels(const AudioBuffer& ab, size_t num_channels) {
  EXPECT_EQ(ab.num_channels(), num_channels);
}

}  // namespace

TEST(AudioBufferTest, SetNumChannelsSetsChannelBuffersNumChannels) {
  AudioBuffer ab(kSampleRateHz, kStereo, kSampleRateHz, kStereo, kSampleRateHz,
                 kStereo);
  ExpectNumChannels(ab, kStereo);
  ab.set_num_channels(1);
  ExpectNumChannels(ab, kMono);
  ab.RestoreNumChannels();
  ExpectNumChannels(ab, kStereo);
}

#if RTC_DCHECK_IS_ON && GTEST_HAS_DEATH_TEST && !defined(WEBRTC_ANDROID)
TEST(AudioBufferDeathTest, SetNumChannelsDeathTest) {
  AudioBuffer ab(kSampleRateHz, kMono, kSampleRateHz, kMono, kSampleRateHz,
                 kMono);
  RTC_EXPECT_DEATH(ab.set_num_channels(kStereo), "num_channels");
}
#endif

TEST(AudioBufferTest, CopyWithoutResampling) {
  AudioBuffer ab1(32000, 2, 32000, 2, 32000, 2);
  AudioBuffer ab2(32000, 2, 32000, 2, 32000, 2);
  // Fill first buffer.
  for (size_t ch = 0; ch < ab1.num_channels(); ++ch) {
    for (size_t i = 0; i < ab1.num_frames(); ++i) {
      ab1.channels()[ch][i] = i + ch;
    }
  }
  // Copy to second buffer.
  ab1.CopyTo(&ab2);
  // Verify content of second buffer.
  for (size_t ch = 0; ch < ab2.num_channels(); ++ch) {
    for (size_t i = 0; i < ab2.num_frames(); ++i) {
      EXPECT_EQ(ab2.channels()[ch][i], i + ch);
    }
  }
}

TEST(AudioBufferTest, CopyWithResampling) {
  AudioBuffer ab1(32000, 2, 32000, 2, 48000, 2);
  AudioBuffer ab2(48000, 2, 48000, 2, 48000, 2);
  float energy_ab1 = 0.f;
  float energy_ab2 = 0.f;
  const float pi = std::acos(-1.f);
  // Put a sine and compute energy of first buffer.
  for (size_t ch = 0; ch < ab1.num_channels(); ++ch) {
    for (size_t i = 0; i < ab1.num_frames(); ++i) {
      ab1.channels()[ch][i] = std::sin(2 * pi * 100.f / 32000.f * i);
      energy_ab1 += ab1.channels()[ch][i] * ab1.channels()[ch][i];
    }
  }
  // Copy to second buffer.
  ab1.CopyTo(&ab2);
  // Compute energy of second buffer.
  for (size_t ch = 0; ch < ab2.num_channels(); ++ch) {
    for (size_t i = 0; i < ab2.num_frames(); ++i) {
      energy_ab2 += ab2.channels()[ch][i] * ab2.channels()[ch][i];
    }
  }
  // Verify that energies match.
  EXPECT_NEAR(energy_ab1, energy_ab2 * 32000.f / 48000.f, .01f * energy_ab1);
}

TEST(AudioBufferTest, DeinterleavedView) {
  AudioBuffer ab(48000, 2, 48000, 2, 48000, 2);
  // Fill the buffer with data.
  const float pi = std::acos(-1.f);
  float* const* channels = ab.channels();
  for (size_t ch = 0; ch < ab.num_channels(); ++ch) {
    for (size_t i = 0; i < ab.num_frames(); ++i) {
      channels[ch][i] = std::sin(2 * pi * 100.f / 32000.f * i);
    }
  }

  // Verify that the DeinterleavedView correctly maps to channels.
  DeinterleavedView<float> view = ab.view();
  ASSERT_EQ(view.num_channels(), ab.num_channels());
  for (size_t c = 0; c < view.num_channels(); ++c) {
    MonoView<float> channel = view[c];
    EXPECT_EQ(SamplesPerChannel(channel), ab.num_frames());
    for (size_t s = 0; s < SamplesPerChannel(channel); ++s) {
      ASSERT_EQ(channel[s], channels[c][s]);
    }
  }
}

}  // namespace webrtc
