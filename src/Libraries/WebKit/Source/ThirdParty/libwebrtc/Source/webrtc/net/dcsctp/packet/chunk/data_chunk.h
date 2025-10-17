/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 25, 2023.
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
#ifndef NET_DCSCTP_PACKET_CHUNK_DATA_CHUNK_H_
#define NET_DCSCTP_PACKET_CHUNK_DATA_CHUNK_H_
#include <stddef.h>
#include <stdint.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "net/dcsctp/packet/chunk/chunk.h"
#include "net/dcsctp/packet/chunk/data_common.h"
#include "net/dcsctp/packet/data.h"
#include "net/dcsctp/packet/tlv_trait.h"

namespace dcsctp {

// https://tools.ietf.org/html/rfc4960#section-3.3.1
struct DataChunkConfig : ChunkConfig {
  static constexpr int kType = 0;
  static constexpr size_t kHeaderSize = 16;
  static constexpr size_t kVariableLengthAlignment = 1;
};

class DataChunk : public AnyDataChunk, public TLVTrait<DataChunkConfig> {
 public:
  static constexpr int kType = DataChunkConfig::kType;

  // Exposed to allow the retransmission queue to make room for the correct
  // header size.
  static constexpr size_t kHeaderSize = DataChunkConfig::kHeaderSize;

  DataChunk(TSN tsn,
            StreamID stream_id,
            SSN ssn,
            PPID ppid,
            std::vector<uint8_t> payload,
            const Options& options)
      : AnyDataChunk(tsn,
                     stream_id,
                     ssn,
                     MID(0),
                     FSN(0),
                     ppid,
                     std::move(payload),
                     options) {}

  DataChunk(TSN tsn, Data&& data, bool immediate_ack)
      : AnyDataChunk(tsn, std::move(data), immediate_ack) {}

  static std::optional<DataChunk> Parse(rtc::ArrayView<const uint8_t> data);

  void SerializeTo(std::vector<uint8_t>& out) const override;
  std::string ToString() const override;
};

}  // namespace dcsctp

#endif  // NET_DCSCTP_PACKET_CHUNK_DATA_CHUNK_H_
