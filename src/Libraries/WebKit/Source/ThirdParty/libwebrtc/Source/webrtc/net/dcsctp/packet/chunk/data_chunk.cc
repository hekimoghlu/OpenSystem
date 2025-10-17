/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 13, 2023.
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
#include "net/dcsctp/packet/chunk/data_chunk.h"

#include <stdint.h>

#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "api/array_view.h"
#include "net/dcsctp/packet/bounded_byte_reader.h"
#include "net/dcsctp/packet/bounded_byte_writer.h"
#include "net/dcsctp/packet/chunk/data_common.h"
#include "rtc_base/strings/string_builder.h"

namespace dcsctp {

// https://tools.ietf.org/html/rfc4960#section-3.3.1

//   0                   1                   2                   3
//   0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |   Type = 0    | Reserved|U|B|E|    Length                     |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |                              TSN                              |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |      Stream Identifier S      |   Stream Sequence Number n    |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |                  Payload Protocol Identifier                  |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  \                                                               \
//  /                 User Data (seq n of Stream S)                 /
//  \                                                               \
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
constexpr int DataChunk::kType;

std::optional<DataChunk> DataChunk::Parse(rtc::ArrayView<const uint8_t> data) {
  std::optional<BoundedByteReader<kHeaderSize>> reader = ParseTLV(data);
  if (!reader.has_value()) {
    return std::nullopt;
  }

  uint8_t flags = reader->Load8<1>();
  TSN tsn(reader->Load32<4>());
  StreamID stream_identifier(reader->Load16<8>());
  SSN ssn(reader->Load16<10>());
  PPID ppid(reader->Load32<12>());

  Options options;
  options.is_end = Data::IsEnd((flags & (1 << kFlagsBitEnd)) != 0);
  options.is_beginning =
      Data::IsBeginning((flags & (1 << kFlagsBitBeginning)) != 0);
  options.is_unordered = IsUnordered((flags & (1 << kFlagsBitUnordered)) != 0);
  options.immediate_ack =
      ImmediateAckFlag((flags & (1 << kFlagsBitImmediateAck)) != 0);

  return DataChunk(tsn, stream_identifier, ssn, ppid,
                   std::vector<uint8_t>(reader->variable_data().begin(),
                                        reader->variable_data().end()),
                   options);
}

void DataChunk::SerializeTo(std::vector<uint8_t>& out) const {
  BoundedByteWriter<kHeaderSize> writer = AllocateTLV(out, payload().size());

  writer.Store8<1>(
      (*options().is_end ? (1 << kFlagsBitEnd) : 0) |
      (*options().is_beginning ? (1 << kFlagsBitBeginning) : 0) |
      (*options().is_unordered ? (1 << kFlagsBitUnordered) : 0) |
      (*options().immediate_ack ? (1 << kFlagsBitImmediateAck) : 0));
  writer.Store32<4>(*tsn());
  writer.Store16<8>(*stream_id());
  writer.Store16<10>(*ssn());
  writer.Store32<12>(*ppid());

  writer.CopyToVariableData(payload());
}

std::string DataChunk::ToString() const {
  rtc::StringBuilder sb;
  sb << "DATA, type=" << (options().is_unordered ? "unordered" : "ordered")
     << "::"
     << (*options().is_beginning && *options().is_end ? "complete"
         : *options().is_beginning                    ? "first"
         : *options().is_end                          ? "last"
                                                      : "middle")
     << ", tsn=" << *tsn() << ", sid=" << *stream_id() << ", ssn=" << *ssn()
     << ", ppid=" << *ppid() << ", length=" << payload().size();
  return sb.Release();
}

}  // namespace dcsctp
