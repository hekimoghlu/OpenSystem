/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 16, 2022.
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
#include "net/dcsctp/packet/parameter/outgoing_ssn_reset_request_parameter.h"

#include <stddef.h>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "api/array_view.h"
#include "net/dcsctp/common/internal_types.h"
#include "net/dcsctp/packet/bounded_byte_reader.h"
#include "net/dcsctp/packet/bounded_byte_writer.h"
#include "net/dcsctp/packet/tlv_trait.h"
#include "net/dcsctp/public/types.h"
#include "rtc_base/strings/string_builder.h"

namespace dcsctp {

// https://tools.ietf.org/html/rfc6525#section-4.1

//   0                   1                   2                   3
//   0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |     Parameter Type = 13       | Parameter Length = 16 + 2 * N |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |           Re-configuration Request Sequence Number            |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |           Re-configuration Response Sequence Number           |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |                Sender's Last Assigned TSN                     |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |  Stream Number 1 (optional)   |    Stream Number 2 (optional) |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  /                            ......                             /
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |  Stream Number N-1 (optional) |    Stream Number N (optional) |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
constexpr int OutgoingSSNResetRequestParameter::kType;

std::optional<OutgoingSSNResetRequestParameter>
OutgoingSSNResetRequestParameter::Parse(rtc::ArrayView<const uint8_t> data) {
  std::optional<BoundedByteReader<kHeaderSize>> reader = ParseTLV(data);
  if (!reader.has_value()) {
    return std::nullopt;
  }

  ReconfigRequestSN request_sequence_number(reader->Load32<4>());
  ReconfigRequestSN response_sequence_number(reader->Load32<8>());
  TSN sender_last_assigned_tsn(reader->Load32<12>());

  size_t stream_count = reader->variable_data_size() / kStreamIdSize;
  std::vector<StreamID> stream_ids;
  stream_ids.reserve(stream_count);
  for (size_t i = 0; i < stream_count; ++i) {
    BoundedByteReader<kStreamIdSize> sub_reader =
        reader->sub_reader<kStreamIdSize>(i * kStreamIdSize);

    stream_ids.push_back(StreamID(sub_reader.Load16<0>()));
  }

  return OutgoingSSNResetRequestParameter(
      request_sequence_number, response_sequence_number,
      sender_last_assigned_tsn, std::move(stream_ids));
}

void OutgoingSSNResetRequestParameter::SerializeTo(
    std::vector<uint8_t>& out) const {
  size_t variable_size = stream_ids_.size() * kStreamIdSize;
  BoundedByteWriter<kHeaderSize> writer = AllocateTLV(out, variable_size);

  writer.Store32<4>(*request_sequence_number_);
  writer.Store32<8>(*response_sequence_number_);
  writer.Store32<12>(*sender_last_assigned_tsn_);

  for (size_t i = 0; i < stream_ids_.size(); ++i) {
    BoundedByteWriter<kStreamIdSize> sub_writer =
        writer.sub_writer<kStreamIdSize>(i * kStreamIdSize);
    sub_writer.Store16<0>(*stream_ids_[i]);
  }
}

std::string OutgoingSSNResetRequestParameter::ToString() const {
  rtc::StringBuilder sb;
  sb << "Outgoing SSN Reset Request, req_seq_nbr=" << *request_sequence_number()
     << ", resp_seq_nbr=" << *response_sequence_number()
     << ", sender_last_asg_tsn=" << *sender_last_assigned_tsn();
  return sb.Release();
}

}  // namespace dcsctp
