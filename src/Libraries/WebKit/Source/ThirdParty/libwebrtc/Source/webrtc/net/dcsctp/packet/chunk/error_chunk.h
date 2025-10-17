/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 4, 2022.
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
#ifndef NET_DCSCTP_PACKET_CHUNK_ERROR_CHUNK_H_
#define NET_DCSCTP_PACKET_CHUNK_ERROR_CHUNK_H_
#include <stddef.h>
#include <stdint.h>

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "net/dcsctp/packet/chunk/chunk.h"
#include "net/dcsctp/packet/error_cause/error_cause.h"
#include "net/dcsctp/packet/tlv_trait.h"

namespace dcsctp {

// https://tools.ietf.org/html/rfc4960#section-3.3.10
struct ErrorChunkConfig : ChunkConfig {
  static constexpr int kType = 9;
  static constexpr size_t kHeaderSize = 4;
  static constexpr size_t kVariableLengthAlignment = 4;
};

class ErrorChunk : public Chunk, public TLVTrait<ErrorChunkConfig> {
 public:
  static constexpr int kType = ErrorChunkConfig::kType;

  explicit ErrorChunk(Parameters error_causes)
      : error_causes_(std::move(error_causes)) {}

  ErrorChunk(ErrorChunk&& other) = default;
  ErrorChunk& operator=(ErrorChunk&& other) = default;

  static std::optional<ErrorChunk> Parse(rtc::ArrayView<const uint8_t> data);

  void SerializeTo(std::vector<uint8_t>& out) const override;
  std::string ToString() const override;

  const Parameters& error_causes() const { return error_causes_; }

 private:
  Parameters error_causes_;
};

}  // namespace dcsctp

#endif  // NET_DCSCTP_PACKET_CHUNK_ERROR_CHUNK_H_
