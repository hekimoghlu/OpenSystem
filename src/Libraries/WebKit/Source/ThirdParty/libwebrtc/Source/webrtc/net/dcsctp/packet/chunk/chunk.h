/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 23, 2023.
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
#ifndef NET_DCSCTP_PACKET_CHUNK_CHUNK_H_
#define NET_DCSCTP_PACKET_CHUNK_CHUNK_H_

#include <stddef.h>
#include <sys/types.h>

#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "net/dcsctp/packet/data.h"
#include "net/dcsctp/packet/error_cause/error_cause.h"
#include "net/dcsctp/packet/parameter/parameter.h"
#include "net/dcsctp/packet/tlv_trait.h"

namespace dcsctp {

// Base class for all SCTP chunks
class Chunk {
 public:
  Chunk() {}
  virtual ~Chunk() = default;

  // Chunks can contain data payloads that shouldn't be copied unnecessarily.
  Chunk(Chunk&& other) = default;
  Chunk& operator=(Chunk&& other) = default;
  Chunk(const Chunk&) = delete;
  Chunk& operator=(const Chunk&) = delete;

  // Serializes the chunk to `out`, growing it as necessary.
  virtual void SerializeTo(std::vector<uint8_t>& out) const = 0;

  // Returns a human readable description of this chunk and its parameters.
  virtual std::string ToString() const = 0;
};

// Introspects the chunk in `data` and returns a human readable textual
// representation of it, to be used in debugging.
std::string DebugConvertChunkToString(rtc::ArrayView<const uint8_t> data);

struct ChunkConfig {
  static constexpr int kTypeSizeInBytes = 1;
};

}  // namespace dcsctp

#endif  // NET_DCSCTP_PACKET_CHUNK_CHUNK_H_
