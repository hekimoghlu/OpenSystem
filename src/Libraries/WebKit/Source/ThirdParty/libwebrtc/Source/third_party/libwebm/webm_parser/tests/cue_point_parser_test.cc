/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 4, 2021.
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
#include "src/cue_point_parser.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "test_utils/element_parser_test.h"
#include "webm/id.h"

using webm::CuePoint;
using webm::CuePointParser;
using webm::CueTrackPositions;
using webm::ElementParserTest;
using webm::Id;

namespace {

class CuePointParserTest
    : public ElementParserTest<CuePointParser, Id::kCuePoint> {};

TEST_F(CuePointParserTest, DefaultParse) {
  EXPECT_CALL(callback_, OnCuePoint(metadata_, CuePoint{})).Times(1);

  ParseAndVerify();
}

TEST_F(CuePointParserTest, DefaultValues) {
  SetReaderData({
      0xB3,  // ID = 0xB3 (CueTime).
      0x80,  // Size = 0.

      0xB7,  // ID = 0xB7 (CueTrackPositions).
      0x80,  // Size = 0.
  });

  CuePoint cue_point;
  cue_point.time.Set(0, true);
  cue_point.cue_track_positions.emplace_back();
  cue_point.cue_track_positions[0].Set({}, true);

  EXPECT_CALL(callback_, OnCuePoint(metadata_, cue_point)).Times(1);

  ParseAndVerify();
}

TEST_F(CuePointParserTest, CustomValues) {
  SetReaderData({
      0xB3,  // ID = 0xB3 (CueTime).
      0x81,  // Size = 1.
      0x01,  // Body (value = 1).

      0xB7,  // ID = 0xB7 (CueTrackPositions).
      0x83,  // Size = 3.

      0xF1,  //   ID = 0xF1 (CueClusterPosition).
      0x81,  //   Size = 1.
      0x02,  //   Body (value = 2).

      0xB7,  // ID = 0xB7 (CueTrackPositions).
      0x83,  // Size = 3.

      0xF7,  //   ID = 0xF7 (CueTrack).
      0x81,  //   Size = 1.
      0x03,  //   Body (value = 3).
  });

  CuePoint cue_point;
  cue_point.time.Set(1, true);
  CueTrackPositions cue_track_positions;
  cue_track_positions.cluster_position.Set(2, true);
  cue_point.cue_track_positions.emplace_back(cue_track_positions, true);
  cue_track_positions = {};
  cue_track_positions.track.Set(3, true);
  cue_point.cue_track_positions.emplace_back(cue_track_positions, true);

  EXPECT_CALL(callback_, OnCuePoint(metadata_, cue_point)).Times(1);

  ParseAndVerify();
}

}  // namespace
