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
#include "net/dcsctp/packet/parameter/zero_checksum_acceptable_chunk_parameter.h"

#include <stdint.h>

#include <optional>
#include <vector>

#include "api/array_view.h"
#include "rtc_base/strings/string_builder.h"

namespace dcsctp {

// https://www.ietf.org/archive/id/draft-tuexen-tsvwg-sctp-zero-checksum-00.html#section-3

//  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
// |   Type = 0x8001 (suggested)   |          Length = 8           |
// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
// |           Error Detection Method Identifier (EDMID)           |
// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
constexpr int ZeroChecksumAcceptableChunkParameter::kType;

std::optional<ZeroChecksumAcceptableChunkParameter>
ZeroChecksumAcceptableChunkParameter::Parse(
    rtc::ArrayView<const uint8_t> data) {
  std::optional<BoundedByteReader<kHeaderSize>> reader = ParseTLV(data);
  if (!reader.has_value()) {
    return std::nullopt;
  }

  ZeroChecksumAlternateErrorDetectionMethod method(reader->Load32<4>());
  if (method == ZeroChecksumAlternateErrorDetectionMethod::None()) {
    return std::nullopt;
  }
  return ZeroChecksumAcceptableChunkParameter(method);
}

void ZeroChecksumAcceptableChunkParameter::SerializeTo(
    std::vector<uint8_t>& out) const {
  BoundedByteWriter<kHeaderSize> writer = AllocateTLV(out);
  writer.Store32<4>(*error_detection_method_);
}

std::string ZeroChecksumAcceptableChunkParameter::ToString() const {
  rtc::StringBuilder sb;
  sb << "Zero Checksum Acceptable (" << *error_detection_method_ << ")";
  return sb.Release();
}
}  // namespace dcsctp
