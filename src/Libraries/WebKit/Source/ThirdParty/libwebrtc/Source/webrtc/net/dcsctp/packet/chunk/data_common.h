/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 11, 2022.
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
#ifndef NET_DCSCTP_PACKET_CHUNK_DATA_COMMON_H_
#define NET_DCSCTP_PACKET_CHUNK_DATA_COMMON_H_
#include <stdint.h>

#include <utility>
#include <vector>

#include "api/array_view.h"
#include "net/dcsctp/packet/chunk/chunk.h"
#include "net/dcsctp/packet/data.h"

namespace dcsctp {

// Base class for DataChunk and IDataChunk
class AnyDataChunk : public Chunk {
 public:
  // Represents the "immediate ack" flag on DATA/I-DATA, from RFC7053.
  using ImmediateAckFlag = webrtc::StrongAlias<class ImmediateAckFlagTag, bool>;

  // Data chunk options.
  // See https://tools.ietf.org/html/rfc4960#section-3.3.1
  struct Options {
    Data::IsEnd is_end = Data::IsEnd(false);
    Data::IsBeginning is_beginning = Data::IsBeginning(false);
    IsUnordered is_unordered = IsUnordered(false);
    ImmediateAckFlag immediate_ack = ImmediateAckFlag(false);
  };

  TSN tsn() const { return tsn_; }

  Options options() const {
    Options options;
    options.is_end = data_.is_end;
    options.is_beginning = data_.is_beginning;
    options.is_unordered = data_.is_unordered;
    options.immediate_ack = immediate_ack_;
    return options;
  }

  StreamID stream_id() const { return data_.stream_id; }
  SSN ssn() const { return data_.ssn; }
  MID mid() const { return data_.mid; }
  FSN fsn() const { return data_.fsn; }
  PPID ppid() const { return data_.ppid; }
  rtc::ArrayView<const uint8_t> payload() const { return data_.payload; }

  // Extracts the Data from the chunk, as a destructive action.
  Data extract() && { return std::move(data_); }

  AnyDataChunk(TSN tsn,
               StreamID stream_id,
               SSN ssn,
               MID mid,
               FSN fsn,
               PPID ppid,
               std::vector<uint8_t> payload,
               const Options& options)
      : tsn_(tsn),
        data_(stream_id,
              ssn,
              mid,
              fsn,
              ppid,
              std::move(payload),
              options.is_beginning,
              options.is_end,
              options.is_unordered),
        immediate_ack_(options.immediate_ack) {}

  AnyDataChunk(TSN tsn, Data data, bool immediate_ack)
      : tsn_(tsn), data_(std::move(data)), immediate_ack_(immediate_ack) {}

 protected:
  // Bits in `flags` header field.
  static constexpr int kFlagsBitEnd = 0;
  static constexpr int kFlagsBitBeginning = 1;
  static constexpr int kFlagsBitUnordered = 2;
  static constexpr int kFlagsBitImmediateAck = 3;

 private:
  TSN tsn_;
  Data data_;
  ImmediateAckFlag immediate_ack_;
};

}  // namespace dcsctp

#endif  // NET_DCSCTP_PACKET_CHUNK_DATA_COMMON_H_
