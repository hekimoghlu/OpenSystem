/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 5, 2023.
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
#include "modules/audio_processing/test/audio_buffer_tools.h"

#include <string.h>

namespace webrtc {
namespace test {

void SetupFrame(const StreamConfig& stream_config,
                std::vector<float*>* frame,
                std::vector<float>* frame_samples) {
  frame_samples->resize(stream_config.num_channels() *
                        stream_config.num_frames());
  frame->resize(stream_config.num_channels());
  for (size_t ch = 0; ch < stream_config.num_channels(); ++ch) {
    (*frame)[ch] = &(*frame_samples)[ch * stream_config.num_frames()];
  }
}

void CopyVectorToAudioBuffer(const StreamConfig& stream_config,
                             rtc::ArrayView<const float> source,
                             AudioBuffer* destination) {
  std::vector<float*> input;
  std::vector<float> input_samples;

  SetupFrame(stream_config, &input, &input_samples);

  RTC_CHECK_EQ(input_samples.size(), source.size());
  memcpy(input_samples.data(), source.data(),
         source.size() * sizeof(source[0]));

  destination->CopyFrom(&input[0], stream_config);
}

void ExtractVectorFromAudioBuffer(const StreamConfig& stream_config,
                                  AudioBuffer* source,
                                  std::vector<float>* destination) {
  std::vector<float*> output;

  SetupFrame(stream_config, &output, destination);

  source->CopyTo(stream_config, &output[0]);
}

void FillBuffer(float value, AudioBuffer& audio_buffer) {
  for (size_t ch = 0; ch < audio_buffer.num_channels(); ++ch) {
    FillBufferChannel(value, ch, audio_buffer);
  }
}

void FillBufferChannel(float value, int channel, AudioBuffer& audio_buffer) {
  RTC_CHECK_LT(channel, audio_buffer.num_channels());
  for (size_t i = 0; i < audio_buffer.num_frames(); ++i) {
    audio_buffer.channels()[channel][i] = value;
  }
}

}  // namespace test
}  // namespace webrtc
