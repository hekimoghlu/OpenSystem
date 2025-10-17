/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 14, 2022.
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
#include "src/cues_parser.h"

#include "gtest/gtest.h"

#include "test_utils/element_parser_test.h"
#include "webm/id.h"

using webm::CuesParser;
using webm::ElementParserTest;
using webm::Id;

namespace {

class CuesParserTest : public ElementParserTest<CuesParser, Id::kCues> {};

TEST_F(CuesParserTest, DefaultValues) {
  ParseAndVerify();

  SetReaderData({
      0xBB,  // ID = 0xBB (CuePoint).
      0x80,  // Size = 0.
  });
  ParseAndVerify();
}

TEST_F(CuesParserTest, RepeatedValues) {
  SetReaderData({
      0xBB,  // ID = 0xBB (CuePoint).
      0x83,  // Size = 3.

      0xB3,  //   ID = 0xB3 (CueTime).
      0x81,  //   Size = 1.
      0x01,  //   Body (value = 1).

      0xBB,  // ID = 0xBB (CuePoint).
      0x83,  // Size = 3.

      0xB3,  //   ID = 0xB3 (CueTime).
      0x81,  //   Size = 1.
      0x02,  //   Body (value = 2).
  });

  ParseAndVerify();
}

}  // namespace
