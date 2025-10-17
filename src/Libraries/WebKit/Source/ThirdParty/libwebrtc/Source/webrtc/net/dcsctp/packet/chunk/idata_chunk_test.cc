/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 20, 2024.
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
#include "net/dcsctp/packet/chunk/idata_chunk.h"

#include <cstdint>
#include <type_traits>
#include <vector>

#include "api/array_view.h"
#include "net/dcsctp/testing/testing_macros.h"
#include "rtc_base/gunit.h"
#include "test/gmock.h"

namespace dcsctp {
namespace {
using ::testing::ElementsAre;

TEST(IDataChunkTest, AtBeginningFromCapture) {
  /*
  I_DATA chunk(ordered, first segment, TSN: 2487901653, SID: 1, MID: 0,
    payload length: 1180 bytes)
      Chunk type: I_DATA (64)
      Chunk flags: 0x02
      Chunk length: 1200
      Transmission sequence number: 2487901653
      Stream identifier: 0x0001
      Reserved: 0
      Message identifier: 0
      Payload protocol identifier: WebRTC Binary (53)
      Reassembled Message in frame: 39
  */

  uint8_t data[] = {0x40, 0x02, 0x00, 0x15, 0x94, 0x4a, 0x5d, 0xd5,
                    0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x35, 0x01, 0x00, 0x00, 0x00};

  ASSERT_HAS_VALUE_AND_ASSIGN(IDataChunk chunk, IDataChunk::Parse(data));
  EXPECT_EQ(*chunk.tsn(), 2487901653);
  EXPECT_EQ(*chunk.stream_id(), 1);
  EXPECT_EQ(*chunk.mid(), 0u);
  EXPECT_EQ(*chunk.ppid(), 53u);
  EXPECT_EQ(*chunk.fsn(), 0u);  // Not provided (so set to zero)
}

TEST(IDataChunkTest, AtBeginningSerializeAndDeserialize) {
  IDataChunk::Options options;
  options.is_beginning = Data::IsBeginning(true);
  IDataChunk chunk(TSN(123), StreamID(456), MID(789), PPID(53), FSN(0), {1},
                   options);

  std::vector<uint8_t> serialized;
  chunk.SerializeTo(serialized);

  ASSERT_HAS_VALUE_AND_ASSIGN(IDataChunk deserialized,
                              IDataChunk::Parse(serialized));
  EXPECT_EQ(*deserialized.tsn(), 123u);
  EXPECT_EQ(*deserialized.stream_id(), 456u);
  EXPECT_EQ(*deserialized.mid(), 789u);
  EXPECT_EQ(*deserialized.ppid(), 53u);
  EXPECT_EQ(*deserialized.fsn(), 0u);

  EXPECT_EQ(deserialized.ToString(),
            "I-DATA, type=ordered::first, tsn=123, stream_id=456, "
            "mid=789, ppid=53, length=1");
}

TEST(IDataChunkTest, InMiddleFromCapture) {
  /*
  I_DATA chunk(ordered, last segment, TSN: 2487901706, SID: 3, MID: 1,
    FSN: 8, payload length: 560 bytes)
      Chunk type: I_DATA (64)
      Chunk flags: 0x01
      Chunk length: 580
      Transmission sequence number: 2487901706
      Stream identifier: 0x0003
      Reserved: 0
      Message identifier: 1
      Fragment sequence number: 8
      Reassembled SCTP Fragments (10000 bytes, 9 fragments):
  */

  uint8_t data[] = {0x40, 0x01, 0x00, 0x15, 0x94, 0x4a, 0x5e, 0x0a,
                    0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
                    0x00, 0x00, 0x00, 0x08, 0x01, 0x00, 0x00, 0x00};

  ASSERT_HAS_VALUE_AND_ASSIGN(IDataChunk chunk, IDataChunk::Parse(data));
  EXPECT_EQ(*chunk.tsn(), 2487901706);
  EXPECT_EQ(*chunk.stream_id(), 3u);
  EXPECT_EQ(*chunk.mid(), 1u);
  EXPECT_EQ(*chunk.ppid(), 0u);  // Not provided (so set to zero)
  EXPECT_EQ(*chunk.fsn(), 8u);
}

TEST(IDataChunkTest, InMiddleSerializeAndDeserialize) {
  IDataChunk chunk(TSN(123), StreamID(456), MID(789), PPID(0), FSN(101112),
                   {1, 2, 3}, /*options=*/{});

  std::vector<uint8_t> serialized;
  chunk.SerializeTo(serialized);

  ASSERT_HAS_VALUE_AND_ASSIGN(IDataChunk deserialized,
                              IDataChunk::Parse(serialized));
  EXPECT_EQ(*deserialized.tsn(), 123u);
  EXPECT_EQ(*deserialized.stream_id(), 456u);
  EXPECT_EQ(*deserialized.mid(), 789u);
  EXPECT_EQ(*deserialized.ppid(), 0u);
  EXPECT_EQ(*deserialized.fsn(), 101112u);
  EXPECT_THAT(deserialized.payload(), ElementsAre(1, 2, 3));

  EXPECT_EQ(deserialized.ToString(),
            "I-DATA, type=ordered::middle, tsn=123, stream_id=456, "
            "mid=789, fsn=101112, length=3");
}

}  // namespace
}  // namespace dcsctp
