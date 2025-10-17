/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 14, 2025.
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
#include "modules/rtp_rtcp/source/leb128.h"

#include <cstdint>
#include <iterator>
#include <limits>

#include "api/array_view.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

using ::testing::ElementsAre;

TEST(Leb128Test, Size) {
  EXPECT_EQ(Leb128Size(0), 1);
  EXPECT_EQ(Leb128Size(0b0111'1111), 1);
  EXPECT_EQ(Leb128Size(0b1000'0000), 2);
  EXPECT_EQ(Leb128Size(std::numeric_limits<uint64_t>::max()), 10);
}

TEST(Leb128Test, ReadZero) {
  const uint8_t one_byte[] = {0};
  const uint8_t* read_at = one_byte;
  EXPECT_EQ(ReadLeb128(read_at, std::end(one_byte)), uint64_t{0});
  EXPECT_EQ(std::distance(read_at, std::end(one_byte)), 0);
}

TEST(Leb128Test, ReadOneByte) {
  const uint8_t buffer[] = {0b0010'1100};
  const uint8_t* read_at = buffer;
  EXPECT_EQ(ReadLeb128(read_at, std::end(buffer)), uint64_t{0b0010'1100});
  EXPECT_EQ(std::distance(read_at, std::end(buffer)), 0);
}

TEST(Leb128Test, ReadTwoByte) {
  const uint8_t buffer[] = {0b1010'1100, 0b0111'0000};
  const uint8_t* read_at = buffer;
  EXPECT_EQ(ReadLeb128(read_at, std::end(buffer)),
            uint64_t{0b111'0000'010'1100});
  EXPECT_EQ(std::distance(read_at, std::end(buffer)), 0);
}

TEST(Leb128Test, ReadNearlyMaxValue1) {
  const uint8_t buffer[] = {0xff, 0xff, 0xff, 0xff, 0xff,
                            0xff, 0xff, 0xff, 0x7f};
  const uint8_t* read_at = buffer;
  EXPECT_EQ(ReadLeb128(read_at, std::end(buffer)),
            uint64_t{0x7fff'ffff'ffff'ffff});
  EXPECT_EQ(std::distance(read_at, std::end(buffer)), 0);
}

TEST(Leb128Test, ReadNearlyMaxValue2) {
  // This is valid, though not optimal way to store 63 bits of the value.
  const uint8_t buffer[] = {0xff, 0xff, 0xff, 0xff, 0xff,
                            0xff, 0xff, 0xff, 0xff, 0x0};
  const uint8_t* read_at = buffer;
  EXPECT_EQ(ReadLeb128(read_at, std::end(buffer)),
            uint64_t{0x7fff'ffff'ffff'ffff});
  EXPECT_EQ(std::distance(read_at, std::end(buffer)), 0);
}

TEST(Leb128Test, ReadMaxValue) {
  // This is valid, though not optimal way to store 63 bits of the value.
  const uint8_t buffer[] = {0xff, 0xff, 0xff, 0xff, 0xff,
                            0xff, 0xff, 0xff, 0xff, 0x1};
  const uint8_t* read_at = buffer;
  EXPECT_EQ(ReadLeb128(read_at, std::end(buffer)), 0xffff'ffff'ffff'ffff);
  EXPECT_EQ(std::distance(read_at, std::end(buffer)), 0);
}

TEST(Leb128Test, FailsToReadMoreThanMaxValue) {
  const uint8_t buffer[] = {0xff, 0xff, 0xff, 0xff, 0xff,
                            0xff, 0xff, 0xff, 0xff, 0x2};
  const uint8_t* read_at = buffer;
  ReadLeb128(read_at, std::end(buffer));
  EXPECT_EQ(read_at, nullptr);
}

TEST(Leb128Test, DoesntReadMoreThan10Bytes) {
  // Though this array represent leb128 encoded value that can fit in uint64_t,
  // ReadLeb128 function discards it to avoid reading too many bytes from the
  // buffer.
  const uint8_t buffer[] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                            0xff, 0xff, 0xff, 0x80, 0x00};
  const uint8_t* read_at = buffer;
  ReadLeb128(read_at, std::end(buffer));
  EXPECT_EQ(read_at, nullptr);
}

TEST(Leb128Test, WriteZero) {
  uint8_t buffer[16];
  EXPECT_EQ(WriteLeb128(0, buffer), 1);
  EXPECT_EQ(buffer[0], 0);
}

TEST(Leb128Test, WriteOneByteValue) {
  uint8_t buffer[16];
  EXPECT_EQ(WriteLeb128(0b0010'1100, buffer), 1);
  EXPECT_EQ(buffer[0], 0b0010'1100);
}

TEST(Leb128Test, WriteTwoByteValue) {
  uint8_t buffer[16];
  EXPECT_EQ(WriteLeb128(0b11'1111'010'1100, buffer), 2);
  EXPECT_EQ(buffer[0], 0b1010'1100);
  EXPECT_EQ(buffer[1], 0b0011'1111);
}

TEST(Leb128Test, WriteNearlyMaxValue) {
  uint8_t buffer[16];
  EXPECT_EQ(WriteLeb128(0x7fff'ffff'ffff'ffff, buffer), 9);
  EXPECT_THAT(
      rtc::MakeArrayView(buffer, 9),
      ElementsAre(0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f));
}

TEST(Leb128Test, WriteMaxValue) {
  uint8_t buffer[16];
  EXPECT_EQ(WriteLeb128(0xffff'ffff'ffff'ffff, buffer), 10);
  EXPECT_THAT(
      rtc::MakeArrayView(buffer, 10),
      ElementsAre(0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01));
}

}  // namespace
}  // namespace webrtc
