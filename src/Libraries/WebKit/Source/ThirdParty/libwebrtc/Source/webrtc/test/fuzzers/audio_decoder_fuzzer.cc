/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 27, 2022.
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
#include "test/fuzzers/audio_decoder_fuzzer.h"

#include <limits>
#include <optional>

#include "api/audio_codecs/audio_decoder.h"
#include "modules/rtp_rtcp/source/byte_io.h"
#include "rtc_base/checks.h"

namespace webrtc {
namespace {
template <typename T, unsigned int B = sizeof(T)>
bool ParseInt(const uint8_t** data, size_t* remaining_size, T* value) {
  static_assert(std::numeric_limits<T>::is_integer, "Type must be an integer.");
  static_assert(sizeof(T) <= sizeof(uint64_t),
                "Cannot read wider than uint64_t.");
  static_assert(B <= sizeof(T), "T must be at least B bytes wide.");
  if (B > *remaining_size)
    return false;
  uint64_t val = ByteReader<uint64_t, B>::ReadBigEndian(*data);
  *data += B;
  *remaining_size -= B;
  *value = static_cast<T>(val);
  return true;
}
}  // namespace

// This function reads two bytes from the beginning of `data`, interprets them
// as the first packet length, and reads this many bytes if available. The
// payload is inserted into the decoder, and the process continues until no more
// data is available. Either AudioDecoder::Decode or
// AudioDecoder::DecodeRedundant is used, depending on the value of
// `decode_type`.
void FuzzAudioDecoder(DecoderFunctionType decode_type,
                      const uint8_t* data,
                      size_t size,
                      AudioDecoder* decoder,
                      int sample_rate_hz,
                      size_t max_decoded_bytes,
                      int16_t* decoded) {
  const uint8_t* data_ptr = data;
  size_t remaining_size = size;
  size_t packet_len;
  constexpr size_t kMaxNumFuzzedPackets = 200;
  for (size_t num_packets = 0; num_packets < kMaxNumFuzzedPackets;
       ++num_packets) {
    if (!(ParseInt<size_t, 2>(&data_ptr, &remaining_size, &packet_len) &&
          packet_len <= remaining_size)) {
      break;
    }
    AudioDecoder::SpeechType speech_type;
    switch (decode_type) {
      case DecoderFunctionType::kNormalDecode:
        decoder->Decode(data_ptr, packet_len, sample_rate_hz, max_decoded_bytes,
                        decoded, &speech_type);
        break;
      case DecoderFunctionType::kRedundantDecode:
        decoder->DecodeRedundant(data_ptr, packet_len, sample_rate_hz,
                                 max_decoded_bytes, decoded, &speech_type);
        break;
    }
    data_ptr += packet_len;
    remaining_size -= packet_len;
  }
}

}  // namespace webrtc
