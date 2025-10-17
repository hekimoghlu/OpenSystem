/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 1, 2022.
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
#include "modules/audio_coding/codecs/pcm16b/audio_decoder_pcm16b.h"

#include <utility>

#include "modules/audio_coding/codecs/legacy_encoded_audio_frame.h"
#include "modules/audio_coding/codecs/pcm16b/pcm16b.h"
#include "rtc_base/checks.h"

namespace webrtc {

AudioDecoderPcm16B::AudioDecoderPcm16B(int sample_rate_hz, size_t num_channels)
    : sample_rate_hz_(sample_rate_hz), num_channels_(num_channels) {
  RTC_DCHECK(sample_rate_hz == 8000 || sample_rate_hz == 16000 ||
             sample_rate_hz == 32000 || sample_rate_hz == 48000)
      << "Unsupported sample rate " << sample_rate_hz;
  RTC_DCHECK_GE(num_channels, 1);
}

void AudioDecoderPcm16B::Reset() {}

int AudioDecoderPcm16B::SampleRateHz() const {
  return sample_rate_hz_;
}

size_t AudioDecoderPcm16B::Channels() const {
  return num_channels_;
}

int AudioDecoderPcm16B::DecodeInternal(const uint8_t* encoded,
                                       size_t encoded_len,
                                       int sample_rate_hz,
                                       int16_t* decoded,
                                       SpeechType* speech_type) {
  RTC_DCHECK_EQ(sample_rate_hz_, sample_rate_hz);
  // Adjust the encoded length down to ensure the same number of samples in each
  // channel.
  const size_t encoded_len_adjusted =
      PacketDuration(encoded, encoded_len) * 2 *
      Channels();  // 2 bytes per sample per channel
  size_t ret = WebRtcPcm16b_Decode(encoded, encoded_len_adjusted, decoded);
  *speech_type = ConvertSpeechType(1);
  return static_cast<int>(ret);
}

std::vector<AudioDecoder::ParseResult> AudioDecoderPcm16B::ParsePayload(
    rtc::Buffer&& payload,
    uint32_t timestamp) {
  const int samples_per_ms = rtc::CheckedDivExact(sample_rate_hz_, 1000);
  return LegacyEncodedAudioFrame::SplitBySamples(
      this, std::move(payload), timestamp, samples_per_ms * 2 * num_channels_,
      samples_per_ms);
}

int AudioDecoderPcm16B::PacketDuration(const uint8_t* /* encoded */,
                                       size_t encoded_len) const {
  // Two encoded byte per sample per channel.
  return static_cast<int>(encoded_len / (2 * Channels()));
}

int AudioDecoderPcm16B::PacketDurationRedundant(const uint8_t* encoded,
                                                size_t encoded_len) const {
  return PacketDuration(encoded, encoded_len);
}

}  // namespace webrtc
