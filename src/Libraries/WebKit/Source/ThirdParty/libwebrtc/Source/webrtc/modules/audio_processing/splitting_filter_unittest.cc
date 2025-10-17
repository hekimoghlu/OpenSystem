/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 10, 2022.
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

#include "modules/audio_processing/splitting_filter.h"

#include <cmath>

#include "common_audio/channel_buffer.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

const size_t kSamplesPer16kHzChannel = 160;
const size_t kSamplesPer48kHzChannel = 480;

}  // namespace

// Generates a signal from presence or absence of sine waves of different
// frequencies.
// Splits into 3 bands and checks their presence or absence.
// Recombines the bands.
// Calculates the delay.
// Checks that the cross correlation of input and output is high enough at the
// calculated delay.
TEST(SplittingFilterTest, SplitsIntoThreeBandsAndReconstructs) {
  static const int kChannels = 1;
  static const int kSampleRateHz = 48000;
  static const size_t kNumBands = 3;
  static const int kFrequenciesHz[kNumBands] = {1000, 12000, 18000};
  static const float kAmplitude = 8192.f;
  static const size_t kChunks = 8;
  SplittingFilter splitting_filter(kChannels, kNumBands,
                                   kSamplesPer48kHzChannel);
  ChannelBuffer<float> in_data(kSamplesPer48kHzChannel, kChannels, kNumBands);
  ChannelBuffer<float> bands(kSamplesPer48kHzChannel, kChannels, kNumBands);
  ChannelBuffer<float> out_data(kSamplesPer48kHzChannel, kChannels, kNumBands);
  for (size_t i = 0; i < kChunks; ++i) {
    // Input signal generation.
    bool is_present[kNumBands];
    memset(in_data.channels()[0], 0,
           kSamplesPer48kHzChannel * sizeof(in_data.channels()[0][0]));
    for (size_t j = 0; j < kNumBands; ++j) {
      is_present[j] = i & (static_cast<size_t>(1) << j);
      float amplitude = is_present[j] ? kAmplitude : 0.f;
      for (size_t k = 0; k < kSamplesPer48kHzChannel; ++k) {
        in_data.channels()[0][k] +=
            amplitude * sin(2.f * M_PI * kFrequenciesHz[j] *
                            (i * kSamplesPer48kHzChannel + k) / kSampleRateHz);
      }
    }
    // Three band splitting filter.
    splitting_filter.Analysis(&in_data, &bands);
    // Energy calculation.
    float energy[kNumBands];
    for (size_t j = 0; j < kNumBands; ++j) {
      energy[j] = 0.f;
      for (size_t k = 0; k < kSamplesPer16kHzChannel; ++k) {
        energy[j] += bands.channels(j)[0][k] * bands.channels(j)[0][k];
      }
      energy[j] /= kSamplesPer16kHzChannel;
      if (is_present[j]) {
        EXPECT_GT(energy[j], kAmplitude * kAmplitude / 4);
      } else {
        EXPECT_LT(energy[j], kAmplitude * kAmplitude / 4);
      }
    }
    // Three band merge.
    splitting_filter.Synthesis(&bands, &out_data);
    // Delay and cross correlation estimation.
    float xcorr = 0.f;
    for (size_t delay = 0; delay < kSamplesPer48kHzChannel; ++delay) {
      float tmpcorr = 0.f;
      for (size_t j = delay; j < kSamplesPer48kHzChannel; ++j) {
        tmpcorr += in_data.channels()[0][j - delay] * out_data.channels()[0][j];
      }
      tmpcorr /= kSamplesPer48kHzChannel;
      if (tmpcorr > xcorr) {
        xcorr = tmpcorr;
      }
    }
    // High cross correlation check.
    bool any_present = false;
    for (size_t j = 0; j < kNumBands; ++j) {
      any_present |= is_present[j];
    }
    if (any_present) {
      EXPECT_GT(xcorr, kAmplitude * kAmplitude / 4);
    }
  }
}

}  // namespace webrtc
