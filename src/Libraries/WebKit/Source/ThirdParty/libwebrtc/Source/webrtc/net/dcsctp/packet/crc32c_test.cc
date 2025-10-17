/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 13, 2022.
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
#include "net/dcsctp/packet/crc32c.h"

#include "test/gmock.h"

namespace dcsctp {
namespace {

constexpr std::array<const uint8_t, 0> kEmpty = {};
constexpr std::array<const uint8_t, 1> kZero = {0};
constexpr std::array<const uint8_t, 4> kManyZeros = {0, 0, 0, 0};
constexpr std::array<const uint8_t, 4> kShort = {1, 2, 3, 4};
constexpr std::array<const uint8_t, 8> kLong = {1, 2, 3, 4, 5, 6, 7, 8};
// https://tools.ietf.org/html/rfc3720#appendix-B.4
constexpr std::array<const uint8_t, 32> k32Zeros = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
constexpr std::array<const uint8_t, 32> k32Ones = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
constexpr std::array<const uint8_t, 32> k32Incrementing = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
constexpr std::array<const uint8_t, 32> k32Decrementing = {
    31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
    15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0};
constexpr std::array<const uint8_t, 48> kISCSICommandPDU = {
    0x01, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00,
    0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x18, 0x28, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

TEST(Crc32Test, TestVectors) {
  EXPECT_EQ(GenerateCrc32C(kEmpty), 0U);
  EXPECT_EQ(GenerateCrc32C(kZero), 0x51537d52U);
  EXPECT_EQ(GenerateCrc32C(kManyZeros), 0xc74b6748U);
  EXPECT_EQ(GenerateCrc32C(kShort), 0xf48c3029U);
  EXPECT_EQ(GenerateCrc32C(kLong), 0x811f8946U);
  // https://tools.ietf.org/html/rfc3720#appendix-B.4
  EXPECT_EQ(GenerateCrc32C(k32Zeros), 0xaa36918aU);
  EXPECT_EQ(GenerateCrc32C(k32Ones), 0x43aba862U);
  EXPECT_EQ(GenerateCrc32C(k32Incrementing), 0x4e79dd46U);
  EXPECT_EQ(GenerateCrc32C(k32Decrementing), 0x5cdb3f11U);
  EXPECT_EQ(GenerateCrc32C(kISCSICommandPDU), 0x563a96d9U);
}

}  // namespace
}  // namespace dcsctp
