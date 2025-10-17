/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 21, 2022.
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
#include "modules/audio_coding/codecs/legacy_encoded_audio_frame.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "rtc_base/checks.h"

namespace webrtc {

LegacyEncodedAudioFrame::LegacyEncodedAudioFrame(AudioDecoder* decoder,
                                                 rtc::Buffer&& payload)
    : decoder_(decoder), payload_(std::move(payload)) {}

LegacyEncodedAudioFrame::~LegacyEncodedAudioFrame() = default;

size_t LegacyEncodedAudioFrame::Duration() const {
  const int ret = decoder_->PacketDuration(payload_.data(), payload_.size());
  return (ret < 0) ? 0 : static_cast<size_t>(ret);
}

std::optional<AudioDecoder::EncodedAudioFrame::DecodeResult>
LegacyEncodedAudioFrame::Decode(rtc::ArrayView<int16_t> decoded) const {
  AudioDecoder::SpeechType speech_type = AudioDecoder::kSpeech;
  const int ret = decoder_->Decode(
      payload_.data(), payload_.size(), decoder_->SampleRateHz(),
      decoded.size() * sizeof(int16_t), decoded.data(), &speech_type);

  if (ret < 0)
    return std::nullopt;

  return DecodeResult{static_cast<size_t>(ret), speech_type};
}

std::vector<AudioDecoder::ParseResult> LegacyEncodedAudioFrame::SplitBySamples(
    AudioDecoder* decoder,
    rtc::Buffer&& payload,
    uint32_t timestamp,
    size_t bytes_per_ms,
    uint32_t timestamps_per_ms) {
  RTC_DCHECK(payload.data());
  std::vector<AudioDecoder::ParseResult> results;
  size_t split_size_bytes = payload.size();

  // Find a "chunk size" >= 20 ms and < 40 ms.
  const size_t min_chunk_size = bytes_per_ms * 20;
  if (min_chunk_size >= payload.size()) {
    std::unique_ptr<LegacyEncodedAudioFrame> frame(
        new LegacyEncodedAudioFrame(decoder, std::move(payload)));
    results.emplace_back(timestamp, 0, std::move(frame));
  } else {
    // Reduce the split size by half as long as `split_size_bytes` is at least
    // twice the minimum chunk size (so that the resulting size is at least as
    // large as the minimum chunk size).
    while (split_size_bytes >= 2 * min_chunk_size) {
      split_size_bytes /= 2;
    }

    const uint32_t timestamps_per_chunk = static_cast<uint32_t>(
        split_size_bytes * timestamps_per_ms / bytes_per_ms);
    size_t byte_offset;
    uint32_t timestamp_offset;
    for (byte_offset = 0, timestamp_offset = 0; byte_offset < payload.size();
         byte_offset += split_size_bytes,
        timestamp_offset += timestamps_per_chunk) {
      split_size_bytes =
          std::min(split_size_bytes, payload.size() - byte_offset);
      rtc::Buffer new_payload(payload.data() + byte_offset, split_size_bytes);
      std::unique_ptr<LegacyEncodedAudioFrame> frame(
          new LegacyEncodedAudioFrame(decoder, std::move(new_payload)));
      results.emplace_back(timestamp + timestamp_offset, 0, std::move(frame));
    }
  }

  return results;
}

}  // namespace webrtc
