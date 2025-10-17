/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 1, 2023.
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
#include "src/id_element_parser.h"

#include "gtest/gtest.h"

#include "test_utils/element_parser_test.h"
#include "webm/element.h"
#include "webm/id.h"
#include "webm/status.h"

using webm::ElementParserTest;
using webm::Id;
using webm::IdElementParser;
using webm::kUnknownElementSize;
using webm::Status;

namespace {

class IdElementParserTest : public ElementParserTest<IdElementParser> {};

TEST_F(IdElementParserTest, InvalidSize) {
  TestInit(0, Status::kInvalidElementSize);
  TestInit(5, Status::kInvalidElementSize);
  TestInit(kUnknownElementSize, Status::kInvalidElementSize);
}

TEST_F(IdElementParserTest, ValidId) {
  SetReaderData({0xEC});
  ParseAndVerify();
  EXPECT_EQ(Id::kVoid, parser_.value());

  SetReaderData({0x1F, 0x43, 0xB6, 0x75});
  ParseAndVerify();
  EXPECT_EQ(Id::kCluster, parser_.value());
}

TEST_F(IdElementParserTest, IncrementalParse) {
  SetReaderData({0x2A, 0xD7, 0xB1});

  IncrementalParseAndVerify();

  EXPECT_EQ(Id::kTimecodeScale, parser_.value());
}

}  // namespace
