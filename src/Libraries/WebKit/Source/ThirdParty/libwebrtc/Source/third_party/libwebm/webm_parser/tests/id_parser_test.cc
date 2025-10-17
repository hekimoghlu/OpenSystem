/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 26, 2024.
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
#include "src/id_parser.h"

#include "gtest/gtest.h"

#include "test_utils/parser_test.h"
#include "webm/id.h"
#include "webm/status.h"

using webm::Id;
using webm::IdParser;
using webm::ParserTest;
using webm::Status;

namespace {

class IdParserTest : public ParserTest<IdParser> {};

TEST_F(IdParserTest, InvalidId) {
  SetReaderData({0x00});
  ParseAndExpectResult(Status::kInvalidElementId);

  ResetParser();
  SetReaderData({0x0F});
  ParseAndExpectResult(Status::kInvalidElementId);
}

TEST_F(IdParserTest, EarlyEndOfFile) {
  SetReaderData({0x40});
  ParseAndExpectResult(Status::kEndOfFile);
}

TEST_F(IdParserTest, ValidId) {
  SetReaderData({0x80});
  ParseAndVerify();
  EXPECT_EQ(static_cast<Id>(0x80), parser_.id());

  ResetParser();
  SetReaderData({0x1F, 0xFF, 0xFF, 0xFF});
  ParseAndVerify();
  EXPECT_EQ(static_cast<Id>(0x1FFFFFFF), parser_.id());
}

TEST_F(IdParserTest, IncrementalParse) {
  SetReaderData({0x1A, 0x45, 0xDF, 0xA3});
  IncrementalParseAndVerify();
  EXPECT_EQ(Id::kEbml, parser_.id());
}

}  // namespace
