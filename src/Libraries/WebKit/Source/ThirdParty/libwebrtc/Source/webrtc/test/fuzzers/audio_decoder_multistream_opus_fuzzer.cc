/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 25, 2025.
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
#include "api/audio_codecs/opus/audio_decoder_multi_channel_opus.h"
#include "api/audio_codecs/opus/audio_decoder_multi_channel_opus_config.h"
#include "test/fuzzers/audio_decoder_fuzzer.h"

namespace webrtc {

AudioDecoderMultiChannelOpusConfig MakeDecoderConfig(
    int num_channels,
    int num_streams,
    int coupled_streams,
    std::vector<unsigned char> channel_mapping) {
  AudioDecoderMultiChannelOpusConfig config;
  config.num_channels = num_channels;
  config.num_streams = num_streams;
  config.coupled_streams = coupled_streams;
  config.channel_mapping = channel_mapping;
  return config;
}

void FuzzOneInput(const uint8_t* data, size_t size) {
  const std::vector<AudioDecoderMultiChannelOpusConfig> surround_configs = {
      MakeDecoderConfig(1, 1, 0, {0}),  // Mono

      MakeDecoderConfig(2, 2, 0, {0, 0}),  // Copy the first (of
                                           // 2) decoded streams
                                           // into both output
                                           // channel 0 and output
                                           // channel 1. Ignore
                                           // the 2nd decoded
                                           // stream.

      MakeDecoderConfig(4, 2, 2, {0, 1, 2, 3}),             // Quad.
      MakeDecoderConfig(6, 4, 2, {0, 4, 1, 2, 3, 5}),       // 5.1
      MakeDecoderConfig(8, 5, 3, {0, 6, 1, 2, 3, 4, 5, 7})  // 7.1
  };

  const auto config = surround_configs[data[0] % surround_configs.size()];
  RTC_CHECK(config.IsOk());
  std::unique_ptr<AudioDecoder> dec =
      AudioDecoderMultiChannelOpus::MakeAudioDecoder(config);
  RTC_CHECK(dec);
  const int kSampleRateHz = 48000;
  const size_t kAllocatedOuputSizeSamples =
      4 * kSampleRateHz / 10;  // 4x100 ms, 4 times the size of the output array
                               // for the stereo Opus codec. It should be enough
                               // for 8 channels.
  int16_t output[kAllocatedOuputSizeSamples];
  FuzzAudioDecoder(DecoderFunctionType::kNormalDecode, data, size, dec.get(),
                   kSampleRateHz, sizeof(output), output);
}
}  // namespace webrtc
