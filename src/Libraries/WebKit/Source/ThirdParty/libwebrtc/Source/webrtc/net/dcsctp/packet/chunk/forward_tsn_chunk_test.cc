/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 14, 2023.
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
#include "net/dcsctp/packet/chunk/forward_tsn_chunk.h"

#include <stdint.h>

#include <type_traits>
#include <vector>

#include "api/array_view.h"
#include "net/dcsctp/packet/chunk/forward_tsn_common.h"
#include "net/dcsctp/testing/testing_macros.h"
#include "rtc_base/gunit.h"
#include "test/gmock.h"

namespace dcsctp {
namespace {
using ::testing::ElementsAre;

TEST(ForwardTsnChunkTest, FromCapture) {
  /*
  FORWARD_TSN chunk(Cumulative TSN: 1905748778)
      Chunk type: FORWARD_TSN (192)
      Chunk flags: 0x00
      Chunk length: 8
      New cumulative TSN: 1905748778
  */

  uint8_t data[] = {0xc0, 0x00, 0x00, 0x08, 0x71, 0x97, 0x6b, 0x2a};

  ASSERT_HAS_VALUE_AND_ASSIGN(ForwardTsnChunk chunk,
                              ForwardTsnChunk::Parse(data));
  EXPECT_EQ(*chunk.new_cumulative_tsn(), 1905748778u);
}

TEST(ForwardTsnChunkTest, SerializeAndDeserialize) {
  ForwardTsnChunk chunk(
      TSN(123), {ForwardTsnChunk::SkippedStream(StreamID(1), SSN(23)),
                 ForwardTsnChunk::SkippedStream(StreamID(42), SSN(99))});

  std::vector<uint8_t> serialized;
  chunk.SerializeTo(serialized);

  ASSERT_HAS_VALUE_AND_ASSIGN(ForwardTsnChunk deserialized,
                              ForwardTsnChunk::Parse(serialized));
  EXPECT_EQ(*deserialized.new_cumulative_tsn(), 123u);
  EXPECT_THAT(
      deserialized.skipped_streams(),
      ElementsAre(ForwardTsnChunk::SkippedStream(StreamID(1), SSN(23)),
                  ForwardTsnChunk::SkippedStream(StreamID(42), SSN(99))));

  EXPECT_EQ(deserialized.ToString(),
            "FORWARD-TSN, new_cumulative_tsn=123, skip 1:23, skip 42:99");
}

}  // namespace
}  // namespace dcsctp
