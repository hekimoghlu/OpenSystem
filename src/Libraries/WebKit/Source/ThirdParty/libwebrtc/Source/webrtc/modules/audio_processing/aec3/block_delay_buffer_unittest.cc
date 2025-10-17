/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 4, 2024.
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
#include "modules/audio_processing/aec3/block_delay_buffer.h"

#include <string>

#include "modules/audio_processing/aec3/aec3_common.h"
#include "modules/audio_processing/audio_buffer.h"
#include "rtc_base/strings/string_builder.h"
#include "test/gtest.h"

namespace webrtc {

namespace {

float SampleValue(size_t sample_index) {
  return sample_index % 32768;
}

// Populates the frame with linearly increasing sample values for each band.
void PopulateInputFrame(size_t frame_length,
                        size_t num_bands,
                        size_t first_sample_index,
                        float* const* frame) {
  for (size_t k = 0; k < num_bands; ++k) {
    for (size_t i = 0; i < frame_length; ++i) {
      frame[k][i] = SampleValue(first_sample_index + i);
    }
  }
}

std::string ProduceDebugText(int sample_rate_hz, size_t delay) {
  char log_stream_buffer[8 * 1024];
  rtc::SimpleStringBuilder ss(log_stream_buffer);
  ss << "Sample rate: " << sample_rate_hz;
  ss << ", Delay: " << delay;
  return ss.str();
}

}  // namespace

class BlockDelayBufferTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<std::tuple<size_t, int, size_t>> {};

INSTANTIATE_TEST_SUITE_P(
    ParameterCombinations,
    BlockDelayBufferTest,
    ::testing::Combine(::testing::Values(0, 1, 27, 160, 4321, 7021),
                       ::testing::Values(16000, 32000, 48000),
                       ::testing::Values(1, 2, 4)));

// Verifies that the correct signal delay is achived.
TEST_P(BlockDelayBufferTest, CorrectDelayApplied) {
  const size_t delay = std::get<0>(GetParam());
  const int rate = std::get<1>(GetParam());
  const size_t num_channels = std::get<2>(GetParam());

  SCOPED_TRACE(ProduceDebugText(rate, delay));
  size_t num_bands = NumBandsForRate(rate);
  size_t subband_frame_length = 160;

  BlockDelayBuffer delay_buffer(num_channels, num_bands, subband_frame_length,
                                delay);

  static constexpr size_t kNumFramesToProcess = 20;
  for (size_t frame_index = 0; frame_index < kNumFramesToProcess;
       ++frame_index) {
    AudioBuffer audio_buffer(rate, num_channels, rate, num_channels, rate,
                             num_channels);
    if (rate > 16000) {
      audio_buffer.SplitIntoFrequencyBands();
    }
    size_t first_sample_index = frame_index * subband_frame_length;
    for (size_t ch = 0; ch < num_channels; ++ch) {
      PopulateInputFrame(subband_frame_length, num_bands, first_sample_index,
                         &audio_buffer.split_bands(ch)[0]);
    }
    delay_buffer.DelaySignal(&audio_buffer);

    for (size_t ch = 0; ch < num_channels; ++ch) {
      for (size_t band = 0; band < num_bands; ++band) {
        size_t sample_index = first_sample_index;
        for (size_t i = 0; i < subband_frame_length; ++i, ++sample_index) {
          if (sample_index < delay) {
            EXPECT_EQ(0.f, audio_buffer.split_bands(ch)[band][i]);
          } else {
            EXPECT_EQ(SampleValue(sample_index - delay),
                      audio_buffer.split_bands(ch)[band][i]);
          }
        }
      }
    }
  }
}

}  // namespace webrtc
