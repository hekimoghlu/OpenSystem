/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 5, 2022.
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
#include "src/cue_track_positions_parser.h"

#include "gtest/gtest.h"

#include "test_utils/element_parser_test.h"
#include "webm/id.h"

using webm::CueTrackPositions;
using webm::CueTrackPositionsParser;
using webm::ElementParserTest;
using webm::Id;

namespace {

class CueTrackPositionsParserTest
    : public ElementParserTest<CueTrackPositionsParser,
                               Id::kCueTrackPositions> {};

TEST_F(CueTrackPositionsParserTest, DefaultParse) {
  ParseAndVerify();

  const CueTrackPositions cue_track_positions = parser_.value();

  EXPECT_FALSE(cue_track_positions.track.is_present());
  EXPECT_EQ(static_cast<std::uint64_t>(0), cue_track_positions.track.value());

  EXPECT_FALSE(cue_track_positions.cluster_position.is_present());
  EXPECT_EQ(static_cast<std::uint64_t>(0),
            cue_track_positions.cluster_position.value());

  EXPECT_FALSE(cue_track_positions.relative_position.is_present());
  EXPECT_EQ(static_cast<std::uint64_t>(0),
            cue_track_positions.relative_position.value());

  EXPECT_FALSE(cue_track_positions.duration.is_present());
  EXPECT_EQ(static_cast<std::uint64_t>(0),
            cue_track_positions.duration.value());

  EXPECT_FALSE(cue_track_positions.block_number.is_present());
  EXPECT_EQ(static_cast<std::uint64_t>(1),
            cue_track_positions.block_number.value());
}

TEST_F(CueTrackPositionsParserTest, DefaultValues) {
  SetReaderData({
      0xF7,  // ID = 0xF7 (CueTrack).
      0x40, 0x00,  // Size = 0.

      0xF1,  // ID = 0xF1 (CueClusterPosition).
      0x40, 0x00,  // Size = 0.

      0xF0,  // ID = 0xF0 (CueRelativePosition).
      0x40, 0x00,  // Size = 0.

      0xB2,  // ID = 0xB2 (CueDuration).
      0x40, 0x00,  // Size = 0.

      0x53, 0x78,  // ID = 0x5378 (CueBlockNumber).
      0x80,  // Size = 0.
  });

  ParseAndVerify();

  const CueTrackPositions cue_track_positions = parser_.value();

  EXPECT_TRUE(cue_track_positions.track.is_present());
  EXPECT_EQ(static_cast<std::uint64_t>(0), cue_track_positions.track.value());

  EXPECT_TRUE(cue_track_positions.cluster_position.is_present());
  EXPECT_EQ(static_cast<std::uint64_t>(0),
            cue_track_positions.cluster_position.value());

  EXPECT_TRUE(cue_track_positions.relative_position.is_present());
  EXPECT_EQ(static_cast<std::uint64_t>(0),
            cue_track_positions.relative_position.value());

  EXPECT_TRUE(cue_track_positions.duration.is_present());
  EXPECT_EQ(static_cast<std::uint64_t>(0),
            cue_track_positions.duration.value());

  EXPECT_TRUE(cue_track_positions.block_number.is_present());
  EXPECT_EQ(static_cast<std::uint64_t>(1),
            cue_track_positions.block_number.value());
}

TEST_F(CueTrackPositionsParserTest, CustomValues) {
  SetReaderData({
      0xF7,  // ID = 0xF7 (CueTrack).
      0x40, 0x01,  // Size = 1.
      0x01,  // Body (value = 1).

      0xF1,  // ID = 0xF1 (CueClusterPosition).
      0x40, 0x01,  // Size = 1.
      0x02,  // Body (value = 2).

      0xF0,  // ID = 0xF0 (CueRelativePosition).
      0x40, 0x01,  // Size = 1.
      0x03,  // Body (value = 3).

      0xB2,  // ID = 0xB2 (CueDuration).
      0x40, 0x01,  // Size = 1.
      0x04,  // Body (value = 4).

      0x53, 0x78,  // ID = 0x5378 (CueBlockNumber).
      0x81,  // Size = 1.
      0x05,  // Body (value = 5).
  });

  ParseAndVerify();

  const CueTrackPositions cue_track_positions = parser_.value();

  EXPECT_TRUE(cue_track_positions.track.is_present());
  EXPECT_EQ(static_cast<std::uint64_t>(1), cue_track_positions.track.value());

  EXPECT_TRUE(cue_track_positions.cluster_position.is_present());
  EXPECT_EQ(static_cast<std::uint64_t>(2),
            cue_track_positions.cluster_position.value());

  EXPECT_TRUE(cue_track_positions.relative_position.is_present());
  EXPECT_EQ(static_cast<std::uint64_t>(3),
            cue_track_positions.relative_position.value());

  EXPECT_TRUE(cue_track_positions.duration.is_present());
  EXPECT_EQ(static_cast<std::uint64_t>(4),
            cue_track_positions.duration.value());

  EXPECT_TRUE(cue_track_positions.block_number.is_present());
  EXPECT_EQ(static_cast<std::uint64_t>(5),
            cue_track_positions.block_number.value());
}

}  // namespace
