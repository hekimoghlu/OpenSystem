/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 30, 2023.
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
#include "src/time_slice_parser.h"

#include "gtest/gtest.h"

#include "test_utils/element_parser_test.h"
#include "webm/id.h"

using webm::ElementParserTest;
using webm::Id;
using webm::TimeSlice;
using webm::TimeSliceParser;

namespace {

class TimeSliceParserTest
    : public ElementParserTest<TimeSliceParser, Id::kTimeSlice> {};

TEST_F(TimeSliceParserTest, DefaultParse) {
  ParseAndVerify();

  const TimeSlice time_slice = parser_.value();

  EXPECT_FALSE(time_slice.lace_number.is_present());
  EXPECT_EQ(static_cast<std::uint64_t>(0), time_slice.lace_number.value());
}

TEST_F(TimeSliceParserTest, DefaultValues) {
  SetReaderData({
      0xCC,  // ID = 0xCC (LaceNumber).
      0x80,  // Size = 0.
  });

  ParseAndVerify();

  const TimeSlice time_slice = parser_.value();

  EXPECT_TRUE(time_slice.lace_number.is_present());
  EXPECT_EQ(static_cast<std::uint64_t>(0), time_slice.lace_number.value());
}

TEST_F(TimeSliceParserTest, CustomValues) {
  SetReaderData({
      0xCC,  // ID = 0xCC (LaceNumber).
      0x81,  // Size = 1.
      0x01,  // Body (value = 1).
  });

  ParseAndVerify();

  const TimeSlice time_slice = parser_.value();

  EXPECT_TRUE(time_slice.lace_number.is_present());
  EXPECT_EQ(static_cast<std::uint64_t>(1), time_slice.lace_number.value());
}

}  // namespace
