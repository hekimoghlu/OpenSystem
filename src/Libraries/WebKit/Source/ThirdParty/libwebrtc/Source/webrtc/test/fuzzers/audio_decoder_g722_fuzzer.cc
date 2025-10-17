/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 4, 2024.
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
#include "modules/audio_coding/codecs/g722/audio_decoder_g722.h"
#include "test/fuzzers/audio_decoder_fuzzer.h"

namespace webrtc {
void FuzzOneInput(const uint8_t* data, size_t size) {
  if (size > 10000 || size < 1) {
    return;
  }

  std::unique_ptr<AudioDecoder> dec;
  size_t num_channels;
  if (data[0] % 2) {
    dec = std::make_unique<AudioDecoderG722Impl>();
    num_channels = 1;
  } else {
    dec = std::make_unique<AudioDecoderG722StereoImpl>();
    num_channels = 2;
  }
  // Allocate a maximum output size of 100 ms.
  const int sample_rate_hz = dec->SampleRateHz();
  const size_t allocated_ouput_size_samples =
      sample_rate_hz / 10 * num_channels;
  std::unique_ptr<int16_t[]> output =
      std::make_unique<int16_t[]>(allocated_ouput_size_samples);
  FuzzAudioDecoder(
      DecoderFunctionType::kNormalDecode, data, size, dec.get(), sample_rate_hz,
      allocated_ouput_size_samples * sizeof(int16_t), output.get());
}
}  // namespace webrtc
