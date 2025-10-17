/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 17, 2023.
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
#include "net/dcsctp/packet/chunk/abort_chunk.h"

#include <stdint.h>

#include <optional>
#include <utility>
#include <vector>

#include "api/array_view.h"
#include "net/dcsctp/packet/bounded_byte_reader.h"
#include "net/dcsctp/packet/bounded_byte_writer.h"
#include "net/dcsctp/packet/error_cause/error_cause.h"
#include "net/dcsctp/packet/tlv_trait.h"

namespace dcsctp {

// https://tools.ietf.org/html/rfc4960#section-3.3.7

//   0                   1                   2                   3
//   0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |   Type = 6    |Reserved     |T|           Length              |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  \                                                               \
//  /                   zero or more Error Causes                   /
//  \                                                               \
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
constexpr int AbortChunk::kType;

std::optional<AbortChunk> AbortChunk::Parse(
    rtc::ArrayView<const uint8_t> data) {
  std::optional<BoundedByteReader<kHeaderSize>> reader = ParseTLV(data);
  if (!reader.has_value()) {
    return std::nullopt;
  }
  std::optional<Parameters> error_causes =
      Parameters::Parse(reader->variable_data());
  if (!error_causes.has_value()) {
    return std::nullopt;
  }
  uint8_t flags = reader->Load8<1>();
  bool filled_in_verification_tag = (flags & (1 << kFlagsBitT)) == 0;
  return AbortChunk(filled_in_verification_tag, *std::move(error_causes));
}

void AbortChunk::SerializeTo(std::vector<uint8_t>& out) const {
  rtc::ArrayView<const uint8_t> error_causes = error_causes_.data();
  BoundedByteWriter<kHeaderSize> writer = AllocateTLV(out, error_causes.size());
  writer.Store8<1>(filled_in_verification_tag_ ? 0 : (1 << kFlagsBitT));
  writer.CopyToVariableData(error_causes);
}

std::string AbortChunk::ToString() const {
  return "ABORT";
}
}  // namespace dcsctp
