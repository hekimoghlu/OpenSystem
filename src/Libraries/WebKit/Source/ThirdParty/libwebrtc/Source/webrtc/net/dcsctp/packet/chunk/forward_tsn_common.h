/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 17, 2023.
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
#ifndef NET_DCSCTP_PACKET_CHUNK_FORWARD_TSN_COMMON_H_
#define NET_DCSCTP_PACKET_CHUNK_FORWARD_TSN_COMMON_H_
#include <stdint.h>

#include <utility>
#include <vector>

#include "api/array_view.h"
#include "net/dcsctp/packet/chunk/chunk.h"

namespace dcsctp {

// Base class for both ForwardTsnChunk and IForwardTsnChunk
class AnyForwardTsnChunk : public Chunk {
 public:
  struct SkippedStream {
    SkippedStream(StreamID stream_id, SSN ssn)
        : stream_id(stream_id), ssn(ssn), unordered(false), mid(0) {}
    SkippedStream(IsUnordered unordered, StreamID stream_id, MID mid)
        : stream_id(stream_id), ssn(0), unordered(unordered), mid(mid) {}

    StreamID stream_id;

    // Set for FORWARD_TSN
    SSN ssn;

    // Set for I-FORWARD_TSN
    IsUnordered unordered;
    MID mid;

    bool operator==(const SkippedStream& other) const {
      return stream_id == other.stream_id && ssn == other.ssn &&
             unordered == other.unordered && mid == other.mid;
    }
  };

  AnyForwardTsnChunk(TSN new_cumulative_tsn,
                     std::vector<SkippedStream> skipped_streams)
      : new_cumulative_tsn_(new_cumulative_tsn),
        skipped_streams_(std::move(skipped_streams)) {}

  TSN new_cumulative_tsn() const { return new_cumulative_tsn_; }

  rtc::ArrayView<const SkippedStream> skipped_streams() const {
    return skipped_streams_;
  }

 private:
  TSN new_cumulative_tsn_;
  std::vector<SkippedStream> skipped_streams_;
};
}  // namespace dcsctp

#endif  // NET_DCSCTP_PACKET_CHUNK_FORWARD_TSN_COMMON_H_
