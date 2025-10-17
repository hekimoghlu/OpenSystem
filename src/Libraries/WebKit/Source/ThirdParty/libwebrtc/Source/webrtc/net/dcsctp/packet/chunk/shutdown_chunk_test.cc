/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 11, 2021.
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
#include "net/dcsctp/packet/chunk/shutdown_chunk.h"

#include <stdint.h>

#include <type_traits>
#include <vector>

#include "net/dcsctp/testing/testing_macros.h"
#include "rtc_base/gunit.h"

namespace dcsctp {
namespace {
TEST(ShutdownChunkTest, FromCapture) {
  /*
  SHUTDOWN chunk (Cumulative TSN ack: 101831101)
      Chunk type: SHUTDOWN (7)
      Chunk flags: 0x00
      Chunk length: 8
      Cumulative TSN Ack: 101831101
  */

  uint8_t data[] = {0x07, 0x00, 0x00, 0x08, 0x06, 0x11, 0xd1, 0xbd};

  ASSERT_HAS_VALUE_AND_ASSIGN(ShutdownChunk chunk, ShutdownChunk::Parse(data));
  EXPECT_EQ(chunk.cumulative_tsn_ack(), TSN(101831101u));
}

TEST(ShutdownChunkTest, SerializeAndDeserialize) {
  ShutdownChunk chunk(TSN(12345678));

  std::vector<uint8_t> serialized;
  chunk.SerializeTo(serialized);

  ASSERT_HAS_VALUE_AND_ASSIGN(ShutdownChunk deserialized,
                              ShutdownChunk::Parse(serialized));

  EXPECT_EQ(deserialized.cumulative_tsn_ack(), TSN(12345678u));
}

}  // namespace
}  // namespace dcsctp
