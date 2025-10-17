/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 10, 2024.
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
#include "net/dcsctp/packet/chunk/cookie_echo_chunk.h"

#include <stdint.h>

#include <type_traits>
#include <vector>

#include "api/array_view.h"
#include "net/dcsctp/testing/testing_macros.h"
#include "rtc_base/gunit.h"
#include "test/gmock.h"

namespace dcsctp {
namespace {
using ::testing::ElementsAre;

TEST(CookieEchoChunkTest, FromCapture) {
  /*
  COOKIE_ECHO chunk (Cookie length: 256 bytes)
      Chunk type: COOKIE_ECHO (10)
      Chunk flags: 0x00
      Chunk length: 260
      Cookie: 12345678
  */

  uint8_t data[] = {0x0a, 0x00, 0x00, 0x08, 0x12, 0x34, 0x56, 0x78};

  ASSERT_HAS_VALUE_AND_ASSIGN(CookieEchoChunk chunk,
                              CookieEchoChunk::Parse(data));

  EXPECT_THAT(chunk.cookie(), ElementsAre(0x12, 0x34, 0x56, 0x78));
}

TEST(CookieEchoChunkTest, SerializeAndDeserialize) {
  uint8_t cookie[] = {1, 2, 3, 4};
  CookieEchoChunk chunk(cookie);

  std::vector<uint8_t> serialized;
  chunk.SerializeTo(serialized);

  ASSERT_HAS_VALUE_AND_ASSIGN(CookieEchoChunk deserialized,
                              CookieEchoChunk::Parse(serialized));

  EXPECT_THAT(deserialized.cookie(), ElementsAre(1, 2, 3, 4));
  EXPECT_EQ(deserialized.ToString(), "COOKIE-ECHO");
}

}  // namespace
}  // namespace dcsctp
