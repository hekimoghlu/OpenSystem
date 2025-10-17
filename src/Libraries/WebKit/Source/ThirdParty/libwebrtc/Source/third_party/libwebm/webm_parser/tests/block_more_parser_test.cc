/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 25, 2022.
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

// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include "src/block_more_parser.h"

#include "gtest/gtest.h"

#include "test_utils/element_parser_test.h"
#include "webm/id.h"

using webm::BlockMore;
using webm::BlockMoreParser;
using webm::ElementParserTest;
using webm::Id;

namespace {

class BlockMoreParserTest
    : public ElementParserTest<BlockMoreParser, Id::kBlockMore> {};

TEST_F(BlockMoreParserTest, DefaultParse) {
  ParseAndVerify();

  const BlockMore block_more = parser_.value();

  EXPECT_FALSE(block_more.id.is_present());
  EXPECT_EQ(static_cast<std::uint64_t>(1), block_more.id.value());

  EXPECT_FALSE(block_more.data.is_present());
  EXPECT_EQ(std::vector<std::uint8_t>{}, block_more.data.value());
}

TEST_F(BlockMoreParserTest, DefaultValues) {
  SetReaderData({
      0xEE,  // ID = 0xEE (BlockAddID).
      0x80,  // Size = 0.

      0xA5,  // ID = 0xA5 (BlockAdditional).
      0x80,  // Size = 0.
  });

  ParseAndVerify();

  const BlockMore block_more = parser_.value();

  EXPECT_TRUE(block_more.id.is_present());
  EXPECT_EQ(static_cast<std::uint64_t>(1), block_more.id.value());

  EXPECT_TRUE(block_more.data.is_present());
  EXPECT_EQ(std::vector<std::uint8_t>{}, block_more.data.value());
}

TEST_F(BlockMoreParserTest, CustomValues) {
  SetReaderData({
      0xEE,  // ID = 0xEE (BlockAddID).
      0x81,  // Size = 1.
      0x02,  // Body (value = 2).

      0xA5,  // ID = 0xA5 (BlockAdditional).
      0x81,  // Size = 1.
      0x00,  // Body.
  });

  ParseAndVerify();

  const BlockMore block_more = parser_.value();

  EXPECT_TRUE(block_more.id.is_present());
  EXPECT_EQ(static_cast<std::uint64_t>(2), block_more.id.value());

  EXPECT_TRUE(block_more.data.is_present());
  EXPECT_EQ(std::vector<std::uint8_t>{0x00}, block_more.data.value());
}

}  // namespace
