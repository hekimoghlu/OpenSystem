/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 5, 2022.
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
#include "modules/audio_coding/codecs/opus/test/audio_ring_buffer.h"

#include <memory>

#include "common_audio/channel_buffer.h"
#include "test/gtest.h"

namespace webrtc {

class AudioRingBufferTest
    : public ::testing::TestWithParam< ::testing::tuple<int, int, int, int> > {
};

void ReadAndWriteTest(const ChannelBuffer<float>& input,
                      size_t num_write_chunk_frames,
                      size_t num_read_chunk_frames,
                      size_t buffer_frames,
                      ChannelBuffer<float>* output) {
  const size_t num_channels = input.num_channels();
  const size_t total_frames = input.num_frames();
  AudioRingBuffer buf(num_channels, buffer_frames);
  std::unique_ptr<float*[]> slice(new float*[num_channels]);

  size_t input_pos = 0;
  size_t output_pos = 0;
  while (input_pos + buf.WriteFramesAvailable() < total_frames) {
    // Write until the buffer is as full as possible.
    while (buf.WriteFramesAvailable() >= num_write_chunk_frames) {
      buf.Write(input.Slice(slice.get(), input_pos), num_channels,
                num_write_chunk_frames);
      input_pos += num_write_chunk_frames;
    }
    // Read until the buffer is as empty as possible.
    while (buf.ReadFramesAvailable() >= num_read_chunk_frames) {
      EXPECT_LT(output_pos, total_frames);
      buf.Read(output->Slice(slice.get(), output_pos), num_channels,
               num_read_chunk_frames);
      output_pos += num_read_chunk_frames;
    }
  }

  // Write and read the last bit.
  if (input_pos < total_frames) {
    buf.Write(input.Slice(slice.get(), input_pos), num_channels,
              total_frames - input_pos);
  }
  if (buf.ReadFramesAvailable()) {
    buf.Read(output->Slice(slice.get(), output_pos), num_channels,
             buf.ReadFramesAvailable());
  }
  EXPECT_EQ(0u, buf.ReadFramesAvailable());
}

TEST_P(AudioRingBufferTest, ReadDataMatchesWrittenData) {
  const size_t kFrames = 5000;
  const size_t num_channels = ::testing::get<3>(GetParam());

  // Initialize the input data to an increasing sequence.
  ChannelBuffer<float> input(kFrames, static_cast<int>(num_channels));
  for (size_t i = 0; i < num_channels; ++i)
    for (size_t j = 0; j < kFrames; ++j)
      input.channels()[i][j] = (i + 1) * (j + 1);

  ChannelBuffer<float> output(kFrames, static_cast<int>(num_channels));
  ReadAndWriteTest(input, ::testing::get<0>(GetParam()),
                   ::testing::get<1>(GetParam()), ::testing::get<2>(GetParam()),
                   &output);

  // Verify the read data matches the input.
  for (size_t i = 0; i < num_channels; ++i)
    for (size_t j = 0; j < kFrames; ++j)
      EXPECT_EQ(input.channels()[i][j], output.channels()[i][j]);
}

INSTANTIATE_TEST_SUITE_P(
    AudioRingBufferTest,
    AudioRingBufferTest,
    ::testing::Combine(::testing::Values(10, 20, 42),  // num_write_chunk_frames
                       ::testing::Values(1, 10, 17),   // num_read_chunk_frames
                       ::testing::Values(100, 256),    // buffer_frames
                       ::testing::Values(1, 4)));      // num_channels

TEST_F(AudioRingBufferTest, MoveReadPosition) {
  const size_t kNumChannels = 1;
  const float kInputArray[] = {1, 2, 3, 4};
  const size_t kNumFrames = sizeof(kInputArray) / sizeof(*kInputArray);
  ChannelBuffer<float> input(kNumFrames, kNumChannels);
  input.SetDataForTesting(kInputArray, kNumFrames);
  AudioRingBuffer buf(kNumChannels, kNumFrames);
  buf.Write(input.channels(), kNumChannels, kNumFrames);

  buf.MoveReadPositionForward(3);
  ChannelBuffer<float> output(1, kNumChannels);
  buf.Read(output.channels(), kNumChannels, 1);
  EXPECT_EQ(4, output.channels()[0][0]);
  buf.MoveReadPositionBackward(3);
  buf.Read(output.channels(), kNumChannels, 1);
  EXPECT_EQ(2, output.channels()[0][0]);
}

}  // namespace webrtc
