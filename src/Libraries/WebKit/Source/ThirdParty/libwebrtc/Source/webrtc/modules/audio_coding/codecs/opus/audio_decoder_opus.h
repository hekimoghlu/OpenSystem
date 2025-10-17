/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 8, 2024.
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
#ifndef MODULES_AUDIO_CODING_CODECS_OPUS_AUDIO_DECODER_OPUS_H_
#define MODULES_AUDIO_CODING_CODECS_OPUS_AUDIO_DECODER_OPUS_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "api/audio_codecs/audio_decoder.h"
#include "api/field_trials_view.h"
#include "modules/audio_coding/codecs/opus/opus_interface.h"
#include "rtc_base/buffer.h"

namespace webrtc {

class AudioDecoderOpusImpl final : public AudioDecoder {
 public:
  explicit AudioDecoderOpusImpl(const FieldTrialsView& field_trails,
                                size_t num_channels,
                                int sample_rate_hz);

  ~AudioDecoderOpusImpl() override;

  AudioDecoderOpusImpl(const AudioDecoderOpusImpl&) = delete;
  AudioDecoderOpusImpl& operator=(const AudioDecoderOpusImpl&) = delete;

  std::vector<ParseResult> ParsePayload(rtc::Buffer&& payload,
                                        uint32_t timestamp) override;
  void Reset() override;
  int PacketDuration(const uint8_t* encoded, size_t encoded_len) const override;
  int PacketDurationRedundant(const uint8_t* encoded,
                              size_t encoded_len) const override;
  bool PacketHasFec(const uint8_t* encoded, size_t encoded_len) const override;
  int SampleRateHz() const override;
  size_t Channels() const override;
  void GeneratePlc(size_t requested_samples_per_channel,
                   rtc::BufferT<int16_t>* concealment_audio) override;

 protected:
  int DecodeInternal(const uint8_t* encoded,
                     size_t encoded_len,
                     int sample_rate_hz,
                     int16_t* decoded,
                     SpeechType* speech_type) override;
  int DecodeRedundantInternal(const uint8_t* encoded,
                              size_t encoded_len,
                              int sample_rate_hz,
                              int16_t* decoded,
                              SpeechType* speech_type) override;

 private:
  OpusDecInst* dec_state_;
  const size_t channels_;
  const int sample_rate_hz_;
  const bool generate_plc_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_CODECS_OPUS_AUDIO_DECODER_OPUS_H_
