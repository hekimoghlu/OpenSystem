/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 12, 2024.
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
#include "modules/audio_processing/aec3/render_delay_buffer.h"

#include <memory>
#include <string>
#include <vector>

#include "api/array_view.h"
#include "modules/audio_processing/aec3/aec3_common.h"
#include "modules/audio_processing/logging/apm_data_dumper.h"
#include "rtc_base/random.h"
#include "rtc_base/strings/string_builder.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

std::string ProduceDebugText(int sample_rate_hz) {
  rtc::StringBuilder ss;
  ss << "Sample rate: " << sample_rate_hz;
  return ss.Release();
}

}  // namespace

// Verifies that the buffer overflow is correctly reported.
TEST(RenderDelayBuffer, BufferOverflow) {
  const EchoCanceller3Config config;
  for (auto num_channels : {1, 2, 8}) {
    for (auto rate : {16000, 32000, 48000}) {
      SCOPED_TRACE(ProduceDebugText(rate));
      std::unique_ptr<RenderDelayBuffer> delay_buffer(
          RenderDelayBuffer::Create(config, rate, num_channels));
      Block block_to_insert(NumBandsForRate(rate), num_channels);
      for (size_t k = 0; k < 10; ++k) {
        EXPECT_EQ(RenderDelayBuffer::BufferingEvent::kNone,
                  delay_buffer->Insert(block_to_insert));
      }
      bool overrun_occurred = false;
      for (size_t k = 0; k < 1000; ++k) {
        RenderDelayBuffer::BufferingEvent event =
            delay_buffer->Insert(block_to_insert);
        overrun_occurred =
            overrun_occurred ||
            RenderDelayBuffer::BufferingEvent::kRenderOverrun == event;
      }

      EXPECT_TRUE(overrun_occurred);
    }
  }
}

// Verifies that the check for available block works.
TEST(RenderDelayBuffer, AvailableBlock) {
  constexpr size_t kNumChannels = 1;
  constexpr int kSampleRateHz = 48000;
  constexpr size_t kNumBands = NumBandsForRate(kSampleRateHz);
  std::unique_ptr<RenderDelayBuffer> delay_buffer(RenderDelayBuffer::Create(
      EchoCanceller3Config(), kSampleRateHz, kNumChannels));
  Block input_block(kNumBands, kNumChannels, 1.0f);
  EXPECT_EQ(RenderDelayBuffer::BufferingEvent::kNone,
            delay_buffer->Insert(input_block));
  delay_buffer->PrepareCaptureProcessing();
}

// Verifies the AlignFromDelay method.
TEST(RenderDelayBuffer, AlignFromDelay) {
  EchoCanceller3Config config;
  std::unique_ptr<RenderDelayBuffer> delay_buffer(
      RenderDelayBuffer::Create(config, 16000, 1));
  ASSERT_TRUE(delay_buffer->Delay());
  delay_buffer->Reset();
  size_t initial_internal_delay = 0;
  for (size_t delay = initial_internal_delay;
       delay < initial_internal_delay + 20; ++delay) {
    ASSERT_TRUE(delay_buffer->AlignFromDelay(delay));
    EXPECT_EQ(delay, delay_buffer->Delay());
  }
}

#if RTC_DCHECK_IS_ON && GTEST_HAS_DEATH_TEST && !defined(WEBRTC_ANDROID)

// Verifies the check for feasible delay.
// TODO(peah): Re-enable the test once the issue with memory leaks during DEATH
// tests on test bots has been fixed.
TEST(RenderDelayBufferDeathTest, DISABLED_WrongDelay) {
  std::unique_ptr<RenderDelayBuffer> delay_buffer(
      RenderDelayBuffer::Create(EchoCanceller3Config(), 48000, 1));
  EXPECT_DEATH(delay_buffer->AlignFromDelay(21), "");
}

// Verifies the check for the number of bands in the inserted blocks.
TEST(RenderDelayBufferDeathTest, WrongNumberOfBands) {
  for (auto rate : {16000, 32000, 48000}) {
    for (size_t num_channels : {1, 2, 8}) {
      SCOPED_TRACE(ProduceDebugText(rate));
      std::unique_ptr<RenderDelayBuffer> delay_buffer(RenderDelayBuffer::Create(
          EchoCanceller3Config(), rate, num_channels));
      Block block_to_insert(
          NumBandsForRate(rate < 48000 ? rate + 16000 : 16000), num_channels);
      EXPECT_DEATH(delay_buffer->Insert(block_to_insert), "");
    }
  }
}

// Verifies the check for the number of channels in the inserted blocks.
TEST(RenderDelayBufferDeathTest, WrongNumberOfChannels) {
  for (auto rate : {16000, 32000, 48000}) {
    for (size_t num_channels : {1, 2, 8}) {
      SCOPED_TRACE(ProduceDebugText(rate));
      std::unique_ptr<RenderDelayBuffer> delay_buffer(RenderDelayBuffer::Create(
          EchoCanceller3Config(), rate, num_channels));
      Block block_to_insert(NumBandsForRate(rate), num_channels + 1);
      EXPECT_DEATH(delay_buffer->Insert(block_to_insert), "");
    }
  }
}

#endif

}  // namespace webrtc
