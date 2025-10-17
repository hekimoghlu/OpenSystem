/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 15, 2024.
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
#include "src/recursive_parser.h"

#include <cstdint>

#include "gtest/gtest.h"

#include "src/byte_parser.h"
#include "src/element_parser.h"
#include "test_utils/element_parser_test.h"
#include "webm/element.h"
#include "webm/status.h"

using webm::Callback;
using webm::ElementMetadata;
using webm::ElementParser;
using webm::ElementParserTest;
using webm::Reader;
using webm::RecursiveParser;
using webm::Status;
using webm::StringParser;

namespace {

class FailParser : public ElementParser {
 public:
  explicit FailParser(std::size_t /* max_recursion_depth */) { ADD_FAILURE(); }

  Status Init(const ElementMetadata& /* metadata */,
              std::uint64_t /* max_size */) override {
    ADD_FAILURE();
    return Status(Status::kInvalidElementSize);
  }

  Status Feed(Callback* /* callback */, Reader* /* reader */,
              std::uint64_t* num_bytes_read) override {
    ADD_FAILURE();
    *num_bytes_read = 0;
    return Status(Status::kInvalidElementSize);
  }

  int value() const {
    ADD_FAILURE();
    return 0;
  }

  int* mutable_value() {
    ADD_FAILURE();
    return nullptr;
  }
};

class StringParserWrapper : public StringParser {
 public:
  explicit StringParserWrapper(std::size_t max_recursion_depth) {
    EXPECT_EQ(max_recursion_depth, 24);
  }
};

class RecursiveFailParserTest
    : public ElementParserTest<RecursiveParser<FailParser>> {};

TEST_F(RecursiveFailParserTest, NoConstruction) {
  RecursiveParser<FailParser> parser;
}

class RecursiveStringParserTest
    : public ElementParserTest<RecursiveParser<StringParserWrapper>> {};

TEST_F(RecursiveStringParserTest, ParsesOkay) {
  ParseAndVerify();
  EXPECT_EQ("", parser_.value());

  SetReaderData({0x48, 0x69});  // "Hi".
  ParseAndVerify();
  EXPECT_EQ("Hi", parser_.value());
}

TEST_F(RecursiveStringParserTest, ExceedMaxRecursionDepth) {
  ResetParser(0);
  TestInit(0, Status::kExceededRecursionDepthLimit);
}

}  // namespace
