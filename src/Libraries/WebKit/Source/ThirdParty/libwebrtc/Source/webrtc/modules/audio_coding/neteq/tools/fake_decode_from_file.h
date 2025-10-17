/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 19, 2025.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_TOOLS_FAKE_DECODE_FROM_FILE_H_
#define MODULES_AUDIO_CODING_NETEQ_TOOLS_FAKE_DECODE_FROM_FILE_H_

#include <memory>
#include <optional>

#include "api/array_view.h"
#include "api/audio_codecs/audio_decoder.h"
#include "modules/audio_coding/neteq/tools/input_audio_file.h"

namespace webrtc {
namespace test {
// Provides an AudioDecoder implementation that delivers audio data from a file.
// The "encoded" input should contain information about what RTP timestamp the
// encoding represents, and how many samples the decoder should produce for that
// encoding. A helper method PrepareEncoded is provided to prepare such
// encodings. If packets are missing, as determined from the timestamps, the
// file reading will skip forward to match the loss.
class FakeDecodeFromFile : public AudioDecoder {
 public:
  FakeDecodeFromFile(std::unique_ptr<InputAudioFile> input,
                     int sample_rate_hz,
                     bool stereo)
      : input_(std::move(input)),
        sample_rate_hz_(sample_rate_hz),
        stereo_(stereo) {}

  ~FakeDecodeFromFile() = default;

  std::vector<ParseResult> ParsePayload(rtc::Buffer&& payload,
                                        uint32_t timestamp) override;

  void Reset() override {}

  int SampleRateHz() const override { return sample_rate_hz_; }

  size_t Channels() const override { return stereo_ ? 2 : 1; }

  int DecodeInternal(const uint8_t* encoded,
                     size_t encoded_len,
                     int sample_rate_hz,
                     int16_t* decoded,
                     SpeechType* speech_type) override;

  // Reads `samples` from the input file and writes the results to
  // `destination`. Location in file is determined by `timestamp`.
  void ReadFromFile(uint32_t timestamp, size_t samples, int16_t* destination);

  // Helper method. Writes `timestamp`, `samples` and
  // `original_payload_size_bytes` to `encoded` in a format that the
  // FakeDecodeFromFile decoder will understand. `encoded` must be at least 12
  // bytes long.
  static void PrepareEncoded(uint32_t timestamp,
                             size_t samples,
                             size_t original_payload_size_bytes,
                             rtc::ArrayView<uint8_t> encoded);

 private:
  std::unique_ptr<InputAudioFile> input_;
  std::optional<uint32_t> next_timestamp_from_input_;
  const int sample_rate_hz_;
  const bool stereo_;
};

}  // namespace test
}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_TOOLS_FAKE_DECODE_FROM_FILE_H_
