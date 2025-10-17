/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 13, 2024.
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
#ifndef MODULES_AUDIO_CODING_CODECS_OPUS_AUDIO_DECODER_MULTI_CHANNEL_OPUS_IMPL_H_
#define MODULES_AUDIO_CODING_CODECS_OPUS_AUDIO_DECODER_MULTI_CHANNEL_OPUS_IMPL_H_

#include <stddef.h>

#include <memory>
#include <vector>

#include "api/audio_codecs/audio_decoder.h"
#include "api/audio_codecs/audio_format.h"
#include "api/audio_codecs/opus/audio_decoder_multi_channel_opus_config.h"
#include "modules/audio_coding/codecs/opus/opus_interface.h"
#include "rtc_base/buffer.h"

namespace webrtc {

class AudioDecoderMultiChannelOpusImpl final : public AudioDecoder {
 public:
  static std::unique_ptr<AudioDecoderMultiChannelOpusImpl> MakeAudioDecoder(
      AudioDecoderMultiChannelOpusConfig config);

  ~AudioDecoderMultiChannelOpusImpl() override;

  AudioDecoderMultiChannelOpusImpl(const AudioDecoderMultiChannelOpusImpl&) =
      delete;
  AudioDecoderMultiChannelOpusImpl& operator=(
      const AudioDecoderMultiChannelOpusImpl&) = delete;

  std::vector<ParseResult> ParsePayload(rtc::Buffer&& payload,
                                        uint32_t timestamp) override;
  void Reset() override;
  int PacketDuration(const uint8_t* encoded, size_t encoded_len) const override;
  int PacketDurationRedundant(const uint8_t* encoded,
                              size_t encoded_len) const override;
  bool PacketHasFec(const uint8_t* encoded, size_t encoded_len) const override;
  int SampleRateHz() const override;
  size_t Channels() const override;

  static std::optional<AudioDecoderMultiChannelOpusConfig> SdpToConfig(
      const SdpAudioFormat& format);

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
  AudioDecoderMultiChannelOpusImpl(OpusDecInst* dec_state,
                                   AudioDecoderMultiChannelOpusConfig config);

  OpusDecInst* dec_state_;
  const AudioDecoderMultiChannelOpusConfig config_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_CODECS_OPUS_AUDIO_DECODER_MULTI_CHANNEL_OPUS_IMPL_H_
