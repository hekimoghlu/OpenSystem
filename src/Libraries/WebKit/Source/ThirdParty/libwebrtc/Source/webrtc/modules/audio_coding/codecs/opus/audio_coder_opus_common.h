/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 4, 2023.
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
#ifndef MODULES_AUDIO_CODING_CODECS_OPUS_AUDIO_CODER_OPUS_COMMON_H_
#define MODULES_AUDIO_CODING_CODECS_OPUS_AUDIO_CODER_OPUS_COMMON_H_

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/audio_codecs/audio_decoder.h"
#include "api/audio_codecs/audio_format.h"
#include "rtc_base/string_to_number.h"

namespace webrtc {

std::optional<std::string> GetFormatParameter(const SdpAudioFormat& format,
                                              absl::string_view param);

template <typename T>
std::optional<T> GetFormatParameter(const SdpAudioFormat& format,
                                    absl::string_view param) {
  return rtc::StringToNumber<T>(GetFormatParameter(format, param).value_or(""));
}

template <>
std::optional<std::vector<unsigned char>> GetFormatParameter(
    const SdpAudioFormat& format,
    absl::string_view param);

class OpusFrame : public AudioDecoder::EncodedAudioFrame {
 public:
  OpusFrame(AudioDecoder* decoder,
            rtc::Buffer&& payload,
            bool is_primary_payload)
      : decoder_(decoder),
        payload_(std::move(payload)),
        is_primary_payload_(is_primary_payload) {}

  size_t Duration() const override {
    int ret;
    if (is_primary_payload_) {
      ret = decoder_->PacketDuration(payload_.data(), payload_.size());
    } else {
      ret = decoder_->PacketDurationRedundant(payload_.data(), payload_.size());
    }
    return (ret < 0) ? 0 : static_cast<size_t>(ret);
  }

  bool IsDtxPacket() const override { return payload_.size() <= 2; }

  std::optional<DecodeResult> Decode(
      rtc::ArrayView<int16_t> decoded) const override {
    AudioDecoder::SpeechType speech_type = AudioDecoder::kSpeech;
    int ret;
    if (is_primary_payload_) {
      ret = decoder_->Decode(
          payload_.data(), payload_.size(), decoder_->SampleRateHz(),
          decoded.size() * sizeof(int16_t), decoded.data(), &speech_type);
    } else {
      ret = decoder_->DecodeRedundant(
          payload_.data(), payload_.size(), decoder_->SampleRateHz(),
          decoded.size() * sizeof(int16_t), decoded.data(), &speech_type);
    }

    if (ret < 0)
      return std::nullopt;

    return DecodeResult{static_cast<size_t>(ret), speech_type};
  }

 private:
  AudioDecoder* const decoder_;
  const rtc::Buffer payload_;
  const bool is_primary_payload_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_CODECS_OPUS_AUDIO_CODER_OPUS_COMMON_H_
