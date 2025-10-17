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
#include "modules/include/module_common_types.h"

#include "modules/include/module_common_types_public.h"
#include "test/gtest.h"

namespace webrtc {

TEST(IsNewerSequenceNumber, Equal) {
  EXPECT_FALSE(IsNewerSequenceNumber(0x0001, 0x0001));
}

TEST(IsNewerSequenceNumber, NoWrap) {
  EXPECT_TRUE(IsNewerSequenceNumber(0xFFFF, 0xFFFE));
  EXPECT_TRUE(IsNewerSequenceNumber(0x0001, 0x0000));
  EXPECT_TRUE(IsNewerSequenceNumber(0x0100, 0x00FF));
}

TEST(IsNewerSequenceNumber, ForwardWrap) {
  EXPECT_TRUE(IsNewerSequenceNumber(0x0000, 0xFFFF));
  EXPECT_TRUE(IsNewerSequenceNumber(0x0000, 0xFF00));
  EXPECT_TRUE(IsNewerSequenceNumber(0x00FF, 0xFFFF));
  EXPECT_TRUE(IsNewerSequenceNumber(0x00FF, 0xFF00));
}

TEST(IsNewerSequenceNumber, BackwardWrap) {
  EXPECT_FALSE(IsNewerSequenceNumber(0xFFFF, 0x0000));
  EXPECT_FALSE(IsNewerSequenceNumber(0xFF00, 0x0000));
  EXPECT_FALSE(IsNewerSequenceNumber(0xFFFF, 0x00FF));
  EXPECT_FALSE(IsNewerSequenceNumber(0xFF00, 0x00FF));
}

TEST(IsNewerSequenceNumber, HalfWayApart) {
  EXPECT_TRUE(IsNewerSequenceNumber(0x8000, 0x0000));
  EXPECT_FALSE(IsNewerSequenceNumber(0x0000, 0x8000));
}

TEST(IsNewerTimestamp, Equal) {
  EXPECT_FALSE(IsNewerTimestamp(0x00000001, 0x000000001));
}

TEST(IsNewerTimestamp, NoWrap) {
  EXPECT_TRUE(IsNewerTimestamp(0xFFFFFFFF, 0xFFFFFFFE));
  EXPECT_TRUE(IsNewerTimestamp(0x00000001, 0x00000000));
  EXPECT_TRUE(IsNewerTimestamp(0x00010000, 0x0000FFFF));
}

TEST(IsNewerTimestamp, ForwardWrap) {
  EXPECT_TRUE(IsNewerTimestamp(0x00000000, 0xFFFFFFFF));
  EXPECT_TRUE(IsNewerTimestamp(0x00000000, 0xFFFF0000));
  EXPECT_TRUE(IsNewerTimestamp(0x0000FFFF, 0xFFFFFFFF));
  EXPECT_TRUE(IsNewerTimestamp(0x0000FFFF, 0xFFFF0000));
}

TEST(IsNewerTimestamp, BackwardWrap) {
  EXPECT_FALSE(IsNewerTimestamp(0xFFFFFFFF, 0x00000000));
  EXPECT_FALSE(IsNewerTimestamp(0xFFFF0000, 0x00000000));
  EXPECT_FALSE(IsNewerTimestamp(0xFFFFFFFF, 0x0000FFFF));
  EXPECT_FALSE(IsNewerTimestamp(0xFFFF0000, 0x0000FFFF));
}

TEST(IsNewerTimestamp, HalfWayApart) {
  EXPECT_TRUE(IsNewerTimestamp(0x80000000, 0x00000000));
  EXPECT_FALSE(IsNewerTimestamp(0x00000000, 0x80000000));
}

TEST(LatestSequenceNumber, NoWrap) {
  EXPECT_EQ(0xFFFFu, LatestSequenceNumber(0xFFFF, 0xFFFE));
  EXPECT_EQ(0x0001u, LatestSequenceNumber(0x0001, 0x0000));
  EXPECT_EQ(0x0100u, LatestSequenceNumber(0x0100, 0x00FF));

  EXPECT_EQ(0xFFFFu, LatestSequenceNumber(0xFFFE, 0xFFFF));
  EXPECT_EQ(0x0001u, LatestSequenceNumber(0x0000, 0x0001));
  EXPECT_EQ(0x0100u, LatestSequenceNumber(0x00FF, 0x0100));
}

TEST(LatestSequenceNumber, Wrap) {
  EXPECT_EQ(0x0000u, LatestSequenceNumber(0x0000, 0xFFFF));
  EXPECT_EQ(0x0000u, LatestSequenceNumber(0x0000, 0xFF00));
  EXPECT_EQ(0x00FFu, LatestSequenceNumber(0x00FF, 0xFFFF));
  EXPECT_EQ(0x00FFu, LatestSequenceNumber(0x00FF, 0xFF00));

  EXPECT_EQ(0x0000u, LatestSequenceNumber(0xFFFF, 0x0000));
  EXPECT_EQ(0x0000u, LatestSequenceNumber(0xFF00, 0x0000));
  EXPECT_EQ(0x00FFu, LatestSequenceNumber(0xFFFF, 0x00FF));
  EXPECT_EQ(0x00FFu, LatestSequenceNumber(0xFF00, 0x00FF));
}

TEST(LatestTimestamp, NoWrap) {
  EXPECT_EQ(0xFFFFFFFFu, LatestTimestamp(0xFFFFFFFF, 0xFFFFFFFE));
  EXPECT_EQ(0x00000001u, LatestTimestamp(0x00000001, 0x00000000));
  EXPECT_EQ(0x00010000u, LatestTimestamp(0x00010000, 0x0000FFFF));
}

TEST(LatestTimestamp, Wrap) {
  EXPECT_EQ(0x00000000u, LatestTimestamp(0x00000000, 0xFFFFFFFF));
  EXPECT_EQ(0x00000000u, LatestTimestamp(0x00000000, 0xFFFF0000));
  EXPECT_EQ(0x0000FFFFu, LatestTimestamp(0x0000FFFF, 0xFFFFFFFF));
  EXPECT_EQ(0x0000FFFFu, LatestTimestamp(0x0000FFFF, 0xFFFF0000));

  EXPECT_EQ(0x00000000u, LatestTimestamp(0xFFFFFFFF, 0x00000000));
  EXPECT_EQ(0x00000000u, LatestTimestamp(0xFFFF0000, 0x00000000));
  EXPECT_EQ(0x0000FFFFu, LatestTimestamp(0xFFFFFFFF, 0x0000FFFF));
  EXPECT_EQ(0x0000FFFFu, LatestTimestamp(0xFFFF0000, 0x0000FFFF));
}

}  // namespace webrtc
