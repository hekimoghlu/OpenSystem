/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 8, 2022.
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
#include <memory>

#include "modules/audio_coding/codecs/pcm16b/audio_decoder_pcm16b.h"
#include "test/fuzzers/audio_decoder_fuzzer.h"

namespace webrtc {
void FuzzOneInput(const uint8_t* data, size_t size) {
  if (size > 10000 || size < 2) {
    return;
  }

  int sample_rate_hz;
  switch (data[0] % 4) {
    case 0:
      sample_rate_hz = 8000;
      break;
    case 1:
      sample_rate_hz = 16000;
      break;
    case 2:
      sample_rate_hz = 32000;
      break;
    case 3:
      sample_rate_hz = 48000;
      break;
    default:
      RTC_DCHECK_NOTREACHED();
      return;
  }
  const size_t num_channels = data[1] % 16 + 1;

  // Two first bytes of the data are used. Move forward.
  data += 2;
  size -= 2;

  AudioDecoderPcm16B dec(sample_rate_hz, num_channels);
  // Allocate a maximum output size of 100 ms.
  const size_t allocated_ouput_size_samples =
      sample_rate_hz * num_channels / 10;
  std::unique_ptr<int16_t[]> output =
      std::make_unique<int16_t[]>(allocated_ouput_size_samples);
  FuzzAudioDecoder(
      DecoderFunctionType::kNormalDecode, data, size, &dec, sample_rate_hz,
      allocated_ouput_size_samples * sizeof(int16_t), output.get());
}
}  // namespace webrtc
