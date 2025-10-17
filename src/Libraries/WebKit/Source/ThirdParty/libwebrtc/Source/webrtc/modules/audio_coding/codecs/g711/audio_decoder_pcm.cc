/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 12, 2024.
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
#include "modules/audio_coding/codecs/g711/audio_decoder_pcm.h"

#include <utility>

#include "modules/audio_coding/codecs/g711/g711_interface.h"
#include "modules/audio_coding/codecs/legacy_encoded_audio_frame.h"

namespace webrtc {

void AudioDecoderPcmU::Reset() {}

std::vector<AudioDecoder::ParseResult> AudioDecoderPcmU::ParsePayload(
    rtc::Buffer&& payload,
    uint32_t timestamp) {
  return LegacyEncodedAudioFrame::SplitBySamples(
      this, std::move(payload), timestamp, 8 * num_channels_, 8);
}

int AudioDecoderPcmU::SampleRateHz() const {
  return 8000;
}

size_t AudioDecoderPcmU::Channels() const {
  return num_channels_;
}

int AudioDecoderPcmU::DecodeInternal(const uint8_t* encoded,
                                     size_t encoded_len,
                                     int sample_rate_hz,
                                     int16_t* decoded,
                                     SpeechType* speech_type) {
  RTC_DCHECK_EQ(SampleRateHz(), sample_rate_hz);
  // Adjust the encoded length down to ensure the same number of samples in each
  // channel.
  const size_t encoded_len_adjusted =
      PacketDuration(encoded, encoded_len) *
      Channels();         // 1 byte per sample per channel
  int16_t temp_type = 1;  // Default is speech.
  size_t ret =
      WebRtcG711_DecodeU(encoded, encoded_len_adjusted, decoded, &temp_type);
  *speech_type = ConvertSpeechType(temp_type);
  return static_cast<int>(ret);
}

int AudioDecoderPcmU::PacketDuration(const uint8_t* /* encoded */,
                                     size_t encoded_len) const {
  // One encoded byte per sample per channel.
  return static_cast<int>(encoded_len / Channels());
}

int AudioDecoderPcmU::PacketDurationRedundant(const uint8_t* encoded,
                                              size_t encoded_len) const {
  return PacketDuration(encoded, encoded_len);
}

void AudioDecoderPcmA::Reset() {}

std::vector<AudioDecoder::ParseResult> AudioDecoderPcmA::ParsePayload(
    rtc::Buffer&& payload,
    uint32_t timestamp) {
  return LegacyEncodedAudioFrame::SplitBySamples(
      this, std::move(payload), timestamp, 8 * num_channels_, 8);
}

int AudioDecoderPcmA::SampleRateHz() const {
  return 8000;
}

size_t AudioDecoderPcmA::Channels() const {
  return num_channels_;
}

int AudioDecoderPcmA::DecodeInternal(const uint8_t* encoded,
                                     size_t encoded_len,
                                     int sample_rate_hz,
                                     int16_t* decoded,
                                     SpeechType* speech_type) {
  RTC_DCHECK_EQ(SampleRateHz(), sample_rate_hz);
  // Adjust the encoded length down to ensure the same number of samples in each
  // channel.
  const size_t encoded_len_adjusted =
      PacketDuration(encoded, encoded_len) *
      Channels();         // 1 byte per sample per channel
  int16_t temp_type = 1;  // Default is speech.
  size_t ret =
      WebRtcG711_DecodeA(encoded, encoded_len_adjusted, decoded, &temp_type);
  *speech_type = ConvertSpeechType(temp_type);
  return static_cast<int>(ret);
}

int AudioDecoderPcmA::PacketDuration(const uint8_t* /* encoded */,
                                     size_t encoded_len) const {
  // One encoded byte per sample per channel.
  return static_cast<int>(encoded_len / Channels());
}

int AudioDecoderPcmA::PacketDurationRedundant(const uint8_t* encoded,
                                              size_t encoded_len) const {
  return PacketDuration(encoded, encoded_len);
}

}  // namespace webrtc
