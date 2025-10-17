/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 26, 2023.
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
#ifndef NET_DCSCTP_PACKET_CHUNK_SACK_CHUNK_H_
#define NET_DCSCTP_PACKET_CHUNK_SACK_CHUNK_H_
#include <stddef.h>

#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "net/dcsctp/packet/chunk/chunk.h"
#include "net/dcsctp/packet/tlv_trait.h"

namespace dcsctp {

// https://tools.ietf.org/html/rfc4960#section-3.3.4
struct SackChunkConfig : ChunkConfig {
  static constexpr int kType = 3;
  static constexpr size_t kHeaderSize = 16;
  static constexpr size_t kVariableLengthAlignment = 4;
};

class SackChunk : public Chunk, public TLVTrait<SackChunkConfig> {
 public:
  static constexpr int kType = SackChunkConfig::kType;

  struct GapAckBlock {
    GapAckBlock(uint16_t start, uint16_t end) : start(start), end(end) {}

    uint16_t start;
    uint16_t end;

    bool operator==(const GapAckBlock& other) const {
      return start == other.start && end == other.end;
    }
  };

  SackChunk(TSN cumulative_tsn_ack,
            uint32_t a_rwnd,
            std::vector<GapAckBlock> gap_ack_blocks,
            std::set<TSN> duplicate_tsns)
      : cumulative_tsn_ack_(cumulative_tsn_ack),
        a_rwnd_(a_rwnd),
        gap_ack_blocks_(std::move(gap_ack_blocks)),
        duplicate_tsns_(std::move(duplicate_tsns)) {}
  static std::optional<SackChunk> Parse(rtc::ArrayView<const uint8_t> data);

  void SerializeTo(std::vector<uint8_t>& out) const override;
  std::string ToString() const override;

  TSN cumulative_tsn_ack() const { return cumulative_tsn_ack_; }
  uint32_t a_rwnd() const { return a_rwnd_; }
  rtc::ArrayView<const GapAckBlock> gap_ack_blocks() const {
    return gap_ack_blocks_;
  }
  const std::set<TSN>& duplicate_tsns() const { return duplicate_tsns_; }

 private:
  static constexpr size_t kGapAckBlockSize = 4;
  static constexpr size_t kDupTsnBlockSize = 4;

  const TSN cumulative_tsn_ack_;
  const uint32_t a_rwnd_;
  std::vector<GapAckBlock> gap_ack_blocks_;
  std::set<TSN> duplicate_tsns_;
};
}  // namespace dcsctp

#endif  // NET_DCSCTP_PACKET_CHUNK_SACK_CHUNK_H_
