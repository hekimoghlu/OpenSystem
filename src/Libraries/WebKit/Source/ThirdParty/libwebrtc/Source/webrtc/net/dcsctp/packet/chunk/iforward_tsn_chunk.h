/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 31, 2023.
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
#ifndef NET_DCSCTP_PACKET_CHUNK_IFORWARD_TSN_CHUNK_H_
#define NET_DCSCTP_PACKET_CHUNK_IFORWARD_TSN_CHUNK_H_
#include <stddef.h>
#include <stdint.h>

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "net/dcsctp/packet/chunk/chunk.h"
#include "net/dcsctp/packet/chunk/forward_tsn_common.h"
#include "net/dcsctp/packet/tlv_trait.h"

namespace dcsctp {

// https://tools.ietf.org/html/rfc8260#section-2.3.1
struct IForwardTsnChunkConfig : ChunkConfig {
  static constexpr int kType = 194;
  static constexpr size_t kHeaderSize = 8;
  static constexpr size_t kVariableLengthAlignment = 8;
};

class IForwardTsnChunk : public AnyForwardTsnChunk,
                         public TLVTrait<IForwardTsnChunkConfig> {
 public:
  static constexpr int kType = IForwardTsnChunkConfig::kType;

  IForwardTsnChunk(TSN new_cumulative_tsn,
                   std::vector<SkippedStream> skipped_streams)
      : AnyForwardTsnChunk(new_cumulative_tsn, std::move(skipped_streams)) {}

  static std::optional<IForwardTsnChunk> Parse(
      rtc::ArrayView<const uint8_t> data);

  void SerializeTo(std::vector<uint8_t>& out) const override;
  std::string ToString() const override;

 private:
  static constexpr size_t kSkippedStreamBufferSize = 8;
};
}  // namespace dcsctp

#endif  // NET_DCSCTP_PACKET_CHUNK_IFORWARD_TSN_CHUNK_H_
