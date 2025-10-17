/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 15, 2024.
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
#include "net/dcsctp/packet/chunk/cookie_ack_chunk.h"

#include <stdint.h>

#include <type_traits>
#include <vector>

#include "net/dcsctp/testing/testing_macros.h"
#include "rtc_base/gunit.h"
#include "test/gmock.h"

namespace dcsctp {
namespace {

TEST(CookieAckChunkTest, FromCapture) {
  /*
  COOKIE_ACK chunk
      Chunk type: COOKIE_ACK (11)
      Chunk flags: 0x00
      Chunk length: 4
  */

  uint8_t data[] = {0x0b, 0x00, 0x00, 0x04};

  EXPECT_TRUE(CookieAckChunk::Parse(data).has_value());
}

TEST(CookieAckChunkTest, SerializeAndDeserialize) {
  CookieAckChunk chunk;

  std::vector<uint8_t> serialized;
  chunk.SerializeTo(serialized);

  ASSERT_HAS_VALUE_AND_ASSIGN(CookieAckChunk deserialized,
                              CookieAckChunk::Parse(serialized));
  EXPECT_EQ(deserialized.ToString(), "COOKIE-ACK");
}

}  // namespace
}  // namespace dcsctp
