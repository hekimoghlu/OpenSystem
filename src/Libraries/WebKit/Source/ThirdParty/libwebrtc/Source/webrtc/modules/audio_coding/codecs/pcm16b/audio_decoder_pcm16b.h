/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 20, 2023.
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
#ifndef MODULES_AUDIO_CODING_CODECS_PCM16B_AUDIO_DECODER_PCM16B_H_
#define MODULES_AUDIO_CODING_CODECS_PCM16B_AUDIO_DECODER_PCM16B_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "api/audio_codecs/audio_decoder.h"
#include "rtc_base/buffer.h"

namespace webrtc {

class AudioDecoderPcm16B final : public AudioDecoder {
 public:
  AudioDecoderPcm16B(int sample_rate_hz, size_t num_channels);

  AudioDecoderPcm16B(const AudioDecoderPcm16B&) = delete;
  AudioDecoderPcm16B& operator=(const AudioDecoderPcm16B&) = delete;

  void Reset() override;
  std::vector<ParseResult> ParsePayload(rtc::Buffer&& payload,
                                        uint32_t timestamp) override;
  int PacketDuration(const uint8_t* encoded, size_t encoded_len) const override;
  int PacketDurationRedundant(const uint8_t* encoded,
                              size_t encoded_len) const override;
  int SampleRateHz() const override;
  size_t Channels() const override;

 protected:
  int DecodeInternal(const uint8_t* encoded,
                     size_t encoded_len,
                     int sample_rate_hz,
                     int16_t* decoded,
                     SpeechType* speech_type) override;

 private:
  const int sample_rate_hz_;
  const size_t num_channels_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_CODECS_PCM16B_AUDIO_DECODER_PCM16B_H_
