/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 24, 2022.
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
#include "net/dcsctp/packet/chunk/iforward_tsn_chunk.h"

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

TEST(IForwardTsnChunkTest, FromCapture) {
  /*
  I_FORWARD_TSN chunk(Cumulative TSN: 3094631148)
      Chunk type: I_FORWARD_TSN (194)
      Chunk flags: 0x00
      Chunk length: 16
      New cumulative TSN: 3094631148
      Stream identifier: 1
      Flags: 0x0000
      Message identifier: 2
  */

  uint8_t data[] = {0xc2, 0x00, 0x00, 0x10, 0xb8, 0x74, 0x52, 0xec,
                    0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02};

  ASSERT_HAS_VALUE_AND_ASSIGN(IForwardTsnChunk chunk,
                              IForwardTsnChunk::Parse(data));
  EXPECT_EQ(*chunk.new_cumulative_tsn(), 3094631148u);
  EXPECT_THAT(chunk.skipped_streams(),
              ElementsAre(IForwardTsnChunk::SkippedStream(
                  IsUnordered(false), StreamID(1), MID(2))));
}

TEST(IForwardTsnChunkTest, SerializeAndDeserialize) {
  IForwardTsnChunk chunk(
      TSN(123), {IForwardTsnChunk::SkippedStream(IsUnordered(false),
                                                 StreamID(1), MID(23)),
                 IForwardTsnChunk::SkippedStream(IsUnordered(true),
                                                 StreamID(42), MID(99))});

  std::vector<uint8_t> serialized;
  chunk.SerializeTo(serialized);

  ASSERT_HAS_VALUE_AND_ASSIGN(IForwardTsnChunk deserialized,
                              IForwardTsnChunk::Parse(serialized));
  EXPECT_EQ(*deserialized.new_cumulative_tsn(), 123u);
  EXPECT_THAT(deserialized.skipped_streams(),
              ElementsAre(IForwardTsnChunk::SkippedStream(IsUnordered(false),
                                                          StreamID(1), MID(23)),
                          IForwardTsnChunk::SkippedStream(
                              IsUnordered(true), StreamID(42), MID(99))));

  EXPECT_EQ(deserialized.ToString(), "I-FORWARD-TSN, new_cumulative_tsn=123");
}

}  // namespace
}  // namespace dcsctp
