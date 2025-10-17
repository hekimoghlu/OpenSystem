/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 18, 2023.
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
#include "src/unknown_parser.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "test_utils/element_parser_test.h"
#include "webm/element.h"
#include "webm/status.h"

using testing::NotNull;

using webm::ElementParserTest;
using webm::kUnknownElementSize;
using webm::Status;
using webm::UnknownParser;

namespace {

class UnknownParserTest : public ElementParserTest<UnknownParser> {};

TEST_F(UnknownParserTest, InvalidSize) {
  TestInit(kUnknownElementSize, Status::kIndefiniteUnknownElement);
}

TEST_F(UnknownParserTest, Empty) {
  EXPECT_CALL(callback_, OnUnknownElement(metadata_, NotNull(), NotNull()))
      .Times(1);

  ParseAndVerify();
}

TEST_F(UnknownParserTest, Valid) {
  SetReaderData({0x00, 0x01, 0x02, 0x04});

  EXPECT_CALL(callback_, OnUnknownElement(metadata_, NotNull(), NotNull()))
      .Times(1);

  ParseAndVerify();
}

TEST_F(UnknownParserTest, IncrementalSkip) {
  SetReaderData({0x00, 0x00, 0x00, 0x00, 0x10, 0x10, 0x10, 0x10});

  EXPECT_CALL(callback_, OnUnknownElement(metadata_, NotNull(), NotNull()))
      .Times(8);

  IncrementalParseAndVerify();
}

}  // namespace
