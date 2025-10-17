/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 12, 2025.
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
#include "net/dcsctp/packet/chunk/chunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "api/array_view.h"
#include "net/dcsctp/common/math.h"
#include "net/dcsctp/packet/chunk/abort_chunk.h"
#include "net/dcsctp/packet/chunk/cookie_ack_chunk.h"
#include "net/dcsctp/packet/chunk/cookie_echo_chunk.h"
#include "net/dcsctp/packet/chunk/data_chunk.h"
#include "net/dcsctp/packet/chunk/error_chunk.h"
#include "net/dcsctp/packet/chunk/forward_tsn_chunk.h"
#include "net/dcsctp/packet/chunk/heartbeat_ack_chunk.h"
#include "net/dcsctp/packet/chunk/heartbeat_request_chunk.h"
#include "net/dcsctp/packet/chunk/idata_chunk.h"
#include "net/dcsctp/packet/chunk/iforward_tsn_chunk.h"
#include "net/dcsctp/packet/chunk/init_ack_chunk.h"
#include "net/dcsctp/packet/chunk/init_chunk.h"
#include "net/dcsctp/packet/chunk/reconfig_chunk.h"
#include "net/dcsctp/packet/chunk/sack_chunk.h"
#include "net/dcsctp/packet/chunk/shutdown_ack_chunk.h"
#include "net/dcsctp/packet/chunk/shutdown_chunk.h"
#include "net/dcsctp/packet/chunk/shutdown_complete_chunk.h"
#include "net/dcsctp/packet/tlv_trait.h"

namespace dcsctp {

template <class Chunk>
bool ParseAndPrint(uint8_t chunk_type,
                   rtc::ArrayView<const uint8_t> data,
                   rtc::StringBuilder& sb) {
  if (chunk_type == Chunk::kType) {
    std::optional<Chunk> c = Chunk::Parse(data);
    if (c.has_value()) {
      sb << c->ToString();
    } else {
      sb << "Failed to parse chunk of type " << chunk_type;
    }
    return true;
  }
  return false;
}

std::string DebugConvertChunkToString(rtc::ArrayView<const uint8_t> data) {
  rtc::StringBuilder sb;

  if (data.empty()) {
    sb << "Failed to parse chunk due to empty data";
  } else {
    uint8_t chunk_type = data[0];
    if (!ParseAndPrint<DataChunk>(chunk_type, data, sb) &&
        !ParseAndPrint<InitChunk>(chunk_type, data, sb) &&
        !ParseAndPrint<InitAckChunk>(chunk_type, data, sb) &&
        !ParseAndPrint<SackChunk>(chunk_type, data, sb) &&
        !ParseAndPrint<HeartbeatRequestChunk>(chunk_type, data, sb) &&
        !ParseAndPrint<HeartbeatAckChunk>(chunk_type, data, sb) &&
        !ParseAndPrint<AbortChunk>(chunk_type, data, sb) &&
        !ParseAndPrint<ErrorChunk>(chunk_type, data, sb) &&
        !ParseAndPrint<CookieEchoChunk>(chunk_type, data, sb) &&
        !ParseAndPrint<CookieAckChunk>(chunk_type, data, sb) &&
        !ParseAndPrint<ShutdownChunk>(chunk_type, data, sb) &&
        !ParseAndPrint<ShutdownAckChunk>(chunk_type, data, sb) &&
        !ParseAndPrint<ShutdownCompleteChunk>(chunk_type, data, sb) &&
        !ParseAndPrint<ReConfigChunk>(chunk_type, data, sb) &&
        !ParseAndPrint<ForwardTsnChunk>(chunk_type, data, sb) &&
        !ParseAndPrint<IDataChunk>(chunk_type, data, sb) &&
        !ParseAndPrint<IForwardTsnChunk>(chunk_type, data, sb)) {
      sb << "Unhandled chunk type: " << static_cast<int>(chunk_type);
    }
  }
  return sb.Release();
}
}  // namespace dcsctp
