/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 20, 2025.
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
#include "src/date_parser.h"

#include "gtest/gtest.h"

#include "test_utils/element_parser_test.h"
#include "webm/status.h"

using webm::DateParser;
using webm::ElementParserTest;
using webm::kUnknownElementSize;
using webm::Status;

namespace {

class DateParserTest : public ElementParserTest<DateParser> {};

TEST_F(DateParserTest, InvalidSize) {
  TestInit(4, Status::kInvalidElementSize);
  TestInit(9, Status::kInvalidElementSize);
  TestInit(kUnknownElementSize, Status::kInvalidElementSize);
}

TEST_F(DateParserTest, CustomDefault) {
  ResetParser(-1);

  ParseAndVerify();

  EXPECT_EQ(-1, parser_.value());
}

TEST_F(DateParserTest, ValidDate) {
  ParseAndVerify();
  EXPECT_EQ(0, parser_.value());

  SetReaderData({0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0});
  ParseAndVerify();
  EXPECT_EQ(0x123456789ABCDEF0, parser_.value());
}

TEST_F(DateParserTest, IncrementalParse) {
  SetReaderData({0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10});

  IncrementalParseAndVerify();

  EXPECT_EQ(static_cast<std::int64_t>(0xFEDCBA9876543210), parser_.value());
}

}  // namespace
