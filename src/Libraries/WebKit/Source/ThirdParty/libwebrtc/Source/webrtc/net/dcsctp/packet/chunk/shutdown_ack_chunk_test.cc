/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 11, 2025.
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
#include "net/dcsctp/packet/chunk/shutdown_ack_chunk.h"

#include <stdint.h>

#include <vector>

#include "rtc_base/gunit.h"
#include "test/gmock.h"

namespace dcsctp {
namespace {

TEST(ShutdownAckChunkTest, FromCapture) {
  /*
  SHUTDOWN_ACK chunk
      Chunk type: SHUTDOWN_ACK (8)
      Chunk flags: 0x00
      Chunk length: 4
  */

  uint8_t data[] = {0x08, 0x00, 0x00, 0x04};

  EXPECT_TRUE(ShutdownAckChunk::Parse(data).has_value());
}

TEST(ShutdownAckChunkTest, SerializeAndDeserialize) {
  ShutdownAckChunk chunk;

  std::vector<uint8_t> serialized;
  chunk.SerializeTo(serialized);

  EXPECT_TRUE(ShutdownAckChunk::Parse(serialized).has_value());
}

}  // namespace
}  // namespace dcsctp
