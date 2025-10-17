/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 6, 2024.
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
#ifndef MODULES_AUDIO_CODING_CODECS_G722_AUDIO_DECODER_G722_H_
#define MODULES_AUDIO_CODING_CODECS_G722_AUDIO_DECODER_G722_H_

#include "api/audio_codecs/audio_decoder.h"

typedef struct WebRtcG722DecInst G722DecInst;

namespace webrtc {

class AudioDecoderG722Impl final : public AudioDecoder {
 public:
  AudioDecoderG722Impl();
  ~AudioDecoderG722Impl() override;

  AudioDecoderG722Impl(const AudioDecoderG722Impl&) = delete;
  AudioDecoderG722Impl& operator=(const AudioDecoderG722Impl&) = delete;

  bool HasDecodePlc() const override;
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
  G722DecInst* dec_state_;
};

class AudioDecoderG722StereoImpl final : public AudioDecoder {
 public:
  AudioDecoderG722StereoImpl();
  ~AudioDecoderG722StereoImpl() override;

  AudioDecoderG722StereoImpl(const AudioDecoderG722StereoImpl&) = delete;
  AudioDecoderG722StereoImpl& operator=(const AudioDecoderG722StereoImpl&) =
      delete;

  void Reset() override;
  std::vector<ParseResult> ParsePayload(rtc::Buffer&& payload,
                                        uint32_t timestamp) override;
  int SampleRateHz() const override;
  int PacketDuration(const uint8_t* encoded, size_t encoded_len) const override;
  size_t Channels() const override;

 protected:
  int DecodeInternal(const uint8_t* encoded,
                     size_t encoded_len,
                     int sample_rate_hz,
                     int16_t* decoded,
                     SpeechType* speech_type) override;

 private:
  // Splits the stereo-interleaved payload in `encoded` into separate payloads
  // for left and right channels. The separated payloads are written to
  // `encoded_deinterleaved`, which must hold at least `encoded_len` samples.
  // The left channel starts at offset 0, while the right channel starts at
  // offset encoded_len / 2 into `encoded_deinterleaved`.
  void SplitStereoPacket(const uint8_t* encoded,
                         size_t encoded_len,
                         uint8_t* encoded_deinterleaved);

  G722DecInst* dec_state_left_;
  G722DecInst* dec_state_right_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_CODECS_G722_AUDIO_DECODER_G722_H_
