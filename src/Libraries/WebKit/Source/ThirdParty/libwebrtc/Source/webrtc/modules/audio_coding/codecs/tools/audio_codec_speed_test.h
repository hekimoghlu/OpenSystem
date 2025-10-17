/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 12, 2025.
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
#ifndef MODULES_AUDIO_CODING_CODECS_TOOLS_AUDIO_CODEC_SPEED_TEST_H_
#define MODULES_AUDIO_CODING_CODECS_TOOLS_AUDIO_CODEC_SPEED_TEST_H_

#include <memory>
#include <string>

#include "test/gtest.h"

namespace webrtc {

// Define coding parameter as
// <channels, bit_rate, file_name, extension, if_save_output>.
typedef std::tuple<size_t, int, std::string, std::string, bool> coding_param;

class AudioCodecSpeedTest : public ::testing::TestWithParam<coding_param> {
 protected:
  AudioCodecSpeedTest(int block_duration_ms,
                      int input_sampling_khz,
                      int output_sampling_khz);
  virtual void SetUp();
  virtual void TearDown();

  // EncodeABlock(...) does the following:
  // 1. encodes a block of audio, saved in `in_data`,
  // 2. save the bit stream to `bit_stream` of `max_bytes` bytes in size,
  // 3. assign `encoded_bytes` with the length of the bit stream (in bytes),
  // 4. return the cost of time (in millisecond) spent on actual encoding.
  virtual float EncodeABlock(int16_t* in_data,
                             uint8_t* bit_stream,
                             size_t max_bytes,
                             size_t* encoded_bytes) = 0;

  // DecodeABlock(...) does the following:
  // 1. decodes the bit stream in `bit_stream` with a length of `encoded_bytes`
  // (in bytes),
  // 2. save the decoded audio in `out_data`,
  // 3. return the cost of time (in millisecond) spent on actual decoding.
  virtual float DecodeABlock(const uint8_t* bit_stream,
                             size_t encoded_bytes,
                             int16_t* out_data) = 0;

  // Encoding and decode an audio of `audio_duration` (in seconds) and
  // record the runtime for encoding and decoding separately.
  void EncodeDecode(size_t audio_duration);

  int block_duration_ms_;
  int input_sampling_khz_;
  int output_sampling_khz_;

  // Number of samples-per-channel in a frame.
  size_t input_length_sample_;

  // Expected output number of samples-per-channel in a frame.
  size_t output_length_sample_;

  std::unique_ptr<int16_t[]> in_data_;
  std::unique_ptr<int16_t[]> out_data_;
  size_t data_pointer_;
  size_t loop_length_samples_;
  std::unique_ptr<uint8_t[]> bit_stream_;

  // Maximum number of bytes in output bitstream for a frame of audio.
  size_t max_bytes_;

  size_t encoded_bytes_;
  float encoding_time_ms_;
  float decoding_time_ms_;
  FILE* out_file_;

  size_t channels_;

  // Bit rate is in bit-per-second.
  int bit_rate_;

  std::string in_filename_;

  // Determines whether to save the output to file.
  bool save_out_data_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_CODECS_TOOLS_AUDIO_CODEC_SPEED_TEST_H_
