/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 4, 2023.
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
#ifndef NET_DCSCTP_PACKET_CHUNK_HEARTBEAT_REQUEST_CHUNK_H_
#define NET_DCSCTP_PACKET_CHUNK_HEARTBEAT_REQUEST_CHUNK_H_
#include <stddef.h>
#include <stdint.h>

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "net/dcsctp/packet/chunk/chunk.h"
#include "net/dcsctp/packet/parameter/heartbeat_info_parameter.h"
#include "net/dcsctp/packet/parameter/parameter.h"
#include "net/dcsctp/packet/tlv_trait.h"

namespace dcsctp {
// https://tools.ietf.org/html/rfc4960#section-3.3.5
struct HeartbeatRequestChunkConfig : ChunkConfig {
  static constexpr int kType = 4;
  static constexpr size_t kHeaderSize = 4;
  static constexpr size_t kVariableLengthAlignment = 1;
};

class HeartbeatRequestChunk : public Chunk,
                              public TLVTrait<HeartbeatRequestChunkConfig> {
 public:
  static constexpr int kType = HeartbeatRequestChunkConfig::kType;

  explicit HeartbeatRequestChunk(Parameters parameters)
      : parameters_(std::move(parameters)) {}

  HeartbeatRequestChunk(HeartbeatRequestChunk&& other) = default;
  HeartbeatRequestChunk& operator=(HeartbeatRequestChunk&& other) = default;

  static std::optional<HeartbeatRequestChunk> Parse(
      rtc::ArrayView<const uint8_t> data);

  void SerializeTo(std::vector<uint8_t>& out) const override;
  std::string ToString() const override;

  const Parameters& parameters() const { return parameters_; }
  Parameters extract_parameters() && { return std::move(parameters_); }
  std::optional<HeartbeatInfoParameter> info() const {
    return parameters_.get<HeartbeatInfoParameter>();
  }

 private:
  Parameters parameters_;
};
}  // namespace dcsctp

#endif  // NET_DCSCTP_PACKET_CHUNK_HEARTBEAT_REQUEST_CHUNK_H_
