/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 6, 2024.
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
#ifndef MODULES_AUDIO_CODING_CODECS_G711_AUDIO_DECODER_PCM_H_
#define MODULES_AUDIO_CODING_CODECS_G711_AUDIO_DECODER_PCM_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "api/audio_codecs/audio_decoder.h"
#include "rtc_base/buffer.h"
#include "rtc_base/checks.h"

namespace webrtc {

class AudioDecoderPcmU final : public AudioDecoder {
 public:
  explicit AudioDecoderPcmU(size_t num_channels) : num_channels_(num_channels) {
    RTC_DCHECK_GE(num_channels, 1);
  }

  AudioDecoderPcmU(const AudioDecoderPcmU&) = delete;
  AudioDecoderPcmU& operator=(const AudioDecoderPcmU&) = delete;

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
  const size_t num_channels_;
};

class AudioDecoderPcmA final : public AudioDecoder {
 public:
  explicit AudioDecoderPcmA(size_t num_channels) : num_channels_(num_channels) {
    RTC_DCHECK_GE(num_channels, 1);
  }

  AudioDecoderPcmA(const AudioDecoderPcmA&) = delete;
  AudioDecoderPcmA& operator=(const AudioDecoderPcmA&) = delete;

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
  const size_t num_channels_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_CODECS_G711_AUDIO_DECODER_PCM_H_
