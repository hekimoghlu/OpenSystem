/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 30, 2025.
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
#include "net/dcsctp/packet/chunk/idata_chunk.h"

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

// https://tools.ietf.org/html/rfc8260#section-2.1

//   0                   1                   2                   3
//   0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |   Type = 64   |  Res  |I|U|B|E|       Length = Variable       |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |                              TSN                              |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |        Stream Identifier      |           Reserved            |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |                      Message Identifier                       |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |    Payload Protocol Identifier / Fragment Sequence Number     |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  \                                                               \
//  /                           User Data                           /
//  \                                                               \
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
constexpr int IDataChunk::kType;

std::optional<IDataChunk> IDataChunk::Parse(
    rtc::ArrayView<const uint8_t> data) {
  std::optional<BoundedByteReader<kHeaderSize>> reader = ParseTLV(data);
  if (!reader.has_value()) {
    return std::nullopt;
  }
  uint8_t flags = reader->Load8<1>();
  TSN tsn(reader->Load32<4>());
  StreamID stream_identifier(reader->Load16<8>());
  MID mid(reader->Load32<12>());
  uint32_t ppid_or_fsn = reader->Load32<16>();

  Options options;
  options.is_end = Data::IsEnd((flags & (1 << kFlagsBitEnd)) != 0);
  options.is_beginning =
      Data::IsBeginning((flags & (1 << kFlagsBitBeginning)) != 0);
  options.is_unordered = IsUnordered((flags & (1 << kFlagsBitUnordered)) != 0);
  options.immediate_ack =
      ImmediateAckFlag((flags & (1 << kFlagsBitImmediateAck)) != 0);

  return IDataChunk(tsn, stream_identifier, mid,
                    PPID(options.is_beginning ? ppid_or_fsn : 0),
                    FSN(options.is_beginning ? 0 : ppid_or_fsn),
                    std::vector<uint8_t>(reader->variable_data().begin(),
                                         reader->variable_data().end()),
                    options);
}

void IDataChunk::SerializeTo(std::vector<uint8_t>& out) const {
  BoundedByteWriter<kHeaderSize> writer = AllocateTLV(out, payload().size());

  writer.Store8<1>(
      (*options().is_end ? (1 << kFlagsBitEnd) : 0) |
      (*options().is_beginning ? (1 << kFlagsBitBeginning) : 0) |
      (*options().is_unordered ? (1 << kFlagsBitUnordered) : 0) |
      (*options().immediate_ack ? (1 << kFlagsBitImmediateAck) : 0));
  writer.Store32<4>(*tsn());
  writer.Store16<8>(*stream_id());
  writer.Store32<12>(*mid());
  writer.Store32<16>(options().is_beginning ? *ppid() : *fsn());
  writer.CopyToVariableData(payload());
}

std::string IDataChunk::ToString() const {
  rtc::StringBuilder sb;
  sb << "I-DATA, type=" << (options().is_unordered ? "unordered" : "ordered")
     << "::"
     << (*options().is_beginning && *options().is_end ? "complete"
         : *options().is_beginning                    ? "first"
         : *options().is_end                          ? "last"
                                                      : "middle")
     << ", tsn=" << *tsn() << ", stream_id=" << *stream_id()
     << ", mid=" << *mid();

  if (*options().is_beginning) {
    sb << ", ppid=" << *ppid();
  } else {
    sb << ", fsn=" << *fsn();
  }
  sb << ", length=" << payload().size();
  return sb.Release();
}

}  // namespace dcsctp
