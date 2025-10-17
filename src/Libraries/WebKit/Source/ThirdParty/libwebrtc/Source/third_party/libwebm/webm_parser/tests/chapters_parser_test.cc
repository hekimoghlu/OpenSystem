/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 19, 2021.
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
#include "src/chapters_parser.h"

#include "gtest/gtest.h"

#include "test_utils/element_parser_test.h"
#include "webm/id.h"

using webm::ChaptersParser;
using webm::ElementParserTest;
using webm::Id;

namespace {

class ChaptersParserTest
    : public ElementParserTest<ChaptersParser, Id::kChapters> {};

TEST_F(ChaptersParserTest, DefaultValues) {
  ParseAndVerify();

  SetReaderData({
      0x45, 0xB9,  // ID = 0x45B9 (EditionEntry).
      0x80,  // Size = 0.
  });
  ParseAndVerify();
}

TEST_F(ChaptersParserTest, RepeatedValues) {
  SetReaderData({
      0x45, 0xB9,  // ID = 0x45B9 (EditionEntry).
      0x84,  // Size = 4.

      0x45, 0xBC,  //   ID = 0x45BC (EditionUID).
      0x81,  //   Size = 1.
      0x01,  //   Body (value = 1).

      0x45, 0xB9,  // ID = 0x45B9 (EditionEntry).
      0x84,  // Size = 4.

      0x45, 0xBC,  //   ID = 0x45BC (EditionUID).
      0x81,  //   Size = 1.
      0x02,  //   Body (value = 2).
  });

  ParseAndVerify();
}

}  // namespace
