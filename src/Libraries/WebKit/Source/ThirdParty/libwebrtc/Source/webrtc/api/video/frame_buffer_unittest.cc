/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 3, 2025.
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
#include "api/video/frame_buffer.h"

#include <optional>
#include <vector>

#include "test/fake_encoded_frame.h"
#include "test/gmock.h"
#include "test/gtest.h"
#include "test/scoped_key_value_config.h"

namespace webrtc {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::IsEmpty;
using ::testing::Matches;

MATCHER_P(FrameWithId, id, "") {
  return Matches(Eq(id))(arg->Id());
}

TEST(FrameBuffer3Test, RejectInvalidRefs) {
  test::ScopedKeyValueConfig field_trials;
  FrameBuffer buffer(/*max_frame_slots=*/10, /*max_decode_history=*/100,
                     field_trials);
  // Ref must be less than the id of this frame.
  EXPECT_FALSE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(0).Id(0).Refs({0}).AsLast().Build()));
  EXPECT_THAT(buffer.LastContinuousFrameId(), Eq(std::nullopt));

  // Duplicate ids are also invalid.
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(10).Id(1).AsLast().Build()));
  EXPECT_FALSE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(20).Id(2).Refs({1, 1}).AsLast().Build()));
  EXPECT_THAT(buffer.LastContinuousFrameId(), Eq(1));
}

TEST(FrameBuffer3Test, LastContinuousUpdatesOnInsertedFrames) {
  test::ScopedKeyValueConfig field_trials;
  FrameBuffer buffer(/*max_frame_slots=*/10, /*max_decode_history=*/100,
                     field_trials);
  EXPECT_THAT(buffer.LastContinuousFrameId(), Eq(std::nullopt));
  EXPECT_THAT(buffer.LastContinuousTemporalUnitFrameId(), Eq(std::nullopt));

  EXPECT_TRUE(
      buffer.InsertFrame(test::FakeFrameBuilder().Time(10).Id(1).Build()));
  EXPECT_THAT(buffer.LastContinuousFrameId(), Eq(1));
  EXPECT_THAT(buffer.LastContinuousTemporalUnitFrameId(), Eq(std::nullopt));

  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(10).Id(2).Refs({1}).AsLast().Build()));
  EXPECT_THAT(buffer.LastContinuousFrameId(), Eq(2));
  EXPECT_THAT(buffer.LastContinuousTemporalUnitFrameId(), Eq(2));
}

TEST(FrameBuffer3Test, LastContinuousFrameReordering) {
  test::ScopedKeyValueConfig field_trials;
  FrameBuffer buffer(/*max_frame_slots=*/10, /*max_decode_history=*/100,
                     field_trials);

  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(10).Id(1).AsLast().Build()));
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(30).Id(3).Refs({2}).AsLast().Build()));
  EXPECT_THAT(buffer.LastContinuousFrameId(), Eq(1));

  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(20).Id(2).Refs({1}).AsLast().Build()));
  EXPECT_THAT(buffer.LastContinuousFrameId(), Eq(3));
}

TEST(FrameBuffer3Test, LastContinuousTemporalUnit) {
  test::ScopedKeyValueConfig field_trials;
  FrameBuffer buffer(/*max_frame_slots=*/10, /*max_decode_history=*/100,
                     field_trials);

  EXPECT_TRUE(
      buffer.InsertFrame(test::FakeFrameBuilder().Time(10).Id(1).Build()));
  EXPECT_THAT(buffer.LastContinuousTemporalUnitFrameId(), Eq(std::nullopt));
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(10).Id(2).Refs({1}).AsLast().Build()));
  EXPECT_THAT(buffer.LastContinuousTemporalUnitFrameId(), Eq(2));
}

TEST(FrameBuffer3Test, LastContinuousTemporalUnitReordering) {
  test::ScopedKeyValueConfig field_trials;
  FrameBuffer buffer(/*max_frame_slots=*/10, /*max_decode_history=*/100,
                     field_trials);

  EXPECT_TRUE(
      buffer.InsertFrame(test::FakeFrameBuilder().Time(10).Id(1).Build()));
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(20).Id(3).Refs({1}).Build()));
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(20).Id(4).Refs({2, 3}).AsLast().Build()));
  EXPECT_THAT(buffer.LastContinuousTemporalUnitFrameId(), Eq(std::nullopt));

  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(10).Id(2).Refs({1}).AsLast().Build()));
  EXPECT_THAT(buffer.LastContinuousTemporalUnitFrameId(), Eq(4));
}

TEST(FrameBuffer3Test, NextDecodable) {
  test::ScopedKeyValueConfig field_trials;
  FrameBuffer buffer(/*max_frame_slots=*/10, /*max_decode_history=*/100,
                     field_trials);

  EXPECT_THAT(buffer.DecodableTemporalUnitsInfo(), Eq(std::nullopt));
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(10).Id(1).AsLast().Build()));
  EXPECT_THAT(buffer.DecodableTemporalUnitsInfo()->next_rtp_timestamp, Eq(10U));
}

TEST(FrameBuffer3Test, AdvanceNextDecodableOnExtraction) {
  test::ScopedKeyValueConfig field_trials;
  FrameBuffer buffer(/*max_frame_slots=*/10, /*max_decode_history=*/100,
                     field_trials);

  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(10).Id(1).AsLast().Build()));
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(20).Id(2).AsLast().Build()));
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(30).Id(3).Refs({2}).AsLast().Build()));
  EXPECT_THAT(buffer.DecodableTemporalUnitsInfo()->next_rtp_timestamp, Eq(10U));

  EXPECT_THAT(buffer.ExtractNextDecodableTemporalUnit(),
              ElementsAre(FrameWithId(1)));
  EXPECT_THAT(buffer.DecodableTemporalUnitsInfo()->next_rtp_timestamp, Eq(20U));
  EXPECT_THAT(buffer.ExtractNextDecodableTemporalUnit(),
              ElementsAre(FrameWithId(2)));
  EXPECT_THAT(buffer.DecodableTemporalUnitsInfo()->next_rtp_timestamp, Eq(30U));
  EXPECT_THAT(buffer.ExtractNextDecodableTemporalUnit(),
              ElementsAre(FrameWithId(3)));
}

TEST(FrameBuffer3Test, AdvanceLastDecodableOnExtraction) {
  test::ScopedKeyValueConfig field_trials;
  FrameBuffer buffer(/*max_frame_slots=*/10, /*max_decode_history=*/100,
                     field_trials);

  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(10).Id(1).AsLast().Build()));
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(20).Id(2).Refs({1}).AsLast().Build()));
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(30).Id(3).Refs({1}).AsLast().Build()));
  EXPECT_THAT(buffer.DecodableTemporalUnitsInfo()->last_rtp_timestamp, Eq(10U));

  EXPECT_THAT(buffer.ExtractNextDecodableTemporalUnit(),
              ElementsAre(FrameWithId(1)));
  EXPECT_THAT(buffer.DecodableTemporalUnitsInfo()->last_rtp_timestamp, Eq(30U));
}

TEST(FrameBuffer3Test, FrameUpdatesNextDecodable) {
  test::ScopedKeyValueConfig field_trials;
  FrameBuffer buffer(/*max_frame_slots=*/10, /*max_decode_history=*/100,
                     field_trials);

  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(20).Id(2).AsLast().Build()));
  EXPECT_THAT(buffer.DecodableTemporalUnitsInfo()->next_rtp_timestamp, Eq(20U));

  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(10).Id(1).AsLast().Build()));
  EXPECT_THAT(buffer.DecodableTemporalUnitsInfo()->next_rtp_timestamp, Eq(10U));
}

TEST(FrameBuffer3Test, KeyframeClearsFullBuffer) {
  test::ScopedKeyValueConfig field_trials;
  FrameBuffer buffer(/*max_frame_slots=*/5, /*max_decode_history=*/10,
                     field_trials);
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(10).Id(1).AsLast().Build()));
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(20).Id(2).Refs({1}).AsLast().Build()));
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(30).Id(3).Refs({2}).AsLast().Build()));
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(40).Id(4).Refs({3}).AsLast().Build()));
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(50).Id(5).Refs({4}).AsLast().Build()));
  EXPECT_THAT(buffer.LastContinuousFrameId(), Eq(5));

  // Frame buffer is full
  EXPECT_FALSE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(60).Id(6).Refs({5}).AsLast().Build()));
  EXPECT_THAT(buffer.LastContinuousFrameId(), Eq(5));

  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(70).Id(7).AsLast().Build()));
  EXPECT_THAT(buffer.LastContinuousFrameId(), Eq(7));
}

TEST(FrameBuffer3Test, DropNextDecodableTemporalUnit) {
  test::ScopedKeyValueConfig field_trials;
  FrameBuffer buffer(/*max_frame_slots=*/10, /*max_decode_history=*/100,
                     field_trials);
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(10).Id(1).AsLast().Build()));
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(20).Id(2).Refs({1}).AsLast().Build()));
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(30).Id(3).Refs({1}).AsLast().Build()));

  buffer.ExtractNextDecodableTemporalUnit();
  buffer.DropNextDecodableTemporalUnit();
  EXPECT_THAT(buffer.ExtractNextDecodableTemporalUnit(),
              ElementsAre(FrameWithId(3)));
}

TEST(FrameBuffer3Test, OldFramesAreIgnored) {
  test::ScopedKeyValueConfig field_trials;
  FrameBuffer buffer(/*max_frame_slots=*/10, /*max_decode_history=*/100,
                     field_trials);
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(10).Id(1).AsLast().Build()));
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(20).Id(2).Refs({1}).AsLast().Build()));

  buffer.ExtractNextDecodableTemporalUnit();
  buffer.ExtractNextDecodableTemporalUnit();

  EXPECT_FALSE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(10).Id(1).AsLast().Build()));
  EXPECT_FALSE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(20).Id(2).Refs({1}).AsLast().Build()));
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(30).Id(3).Refs({1}).AsLast().Build()));
  EXPECT_THAT(buffer.ExtractNextDecodableTemporalUnit(),
              ElementsAre(FrameWithId(3)));
}

TEST(FrameBuffer3Test, ReturnFullTemporalUnitKSVC) {
  test::ScopedKeyValueConfig field_trials;
  FrameBuffer buffer(/*max_frame_slots=*/10, /*max_decode_history=*/100,
                     field_trials);
  EXPECT_TRUE(
      buffer.InsertFrame(test::FakeFrameBuilder().Time(10).Id(1).Build()));
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(10).Id(2).Refs({1}).Build()));
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(10).Id(3).Refs({2}).AsLast().Build()));
  EXPECT_THAT(buffer.ExtractNextDecodableTemporalUnit(),
              ElementsAre(FrameWithId(1), FrameWithId(2), FrameWithId(3)));

  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(20).Id(4).Refs({3}).AsLast().Build()));
  EXPECT_THAT(buffer.ExtractNextDecodableTemporalUnit(),
              ElementsAre(FrameWithId(4)));
}

TEST(FrameBuffer3Test, InterleavedStream) {
  test::ScopedKeyValueConfig field_trials;
  FrameBuffer buffer(/*max_frame_slots=*/10, /*max_decode_history=*/100,
                     field_trials);
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(10).Id(1).AsLast().Build()));
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(20).Id(2).Refs({1}).AsLast().Build()));
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(30).Id(3).Refs({1}).AsLast().Build()));
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(40).Id(4).Refs({2}).AsLast().Build()));
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(50).Id(5).Refs({3}).AsLast().Build()));

  EXPECT_THAT(buffer.ExtractNextDecodableTemporalUnit(),
              ElementsAre(FrameWithId(1)));
  EXPECT_THAT(buffer.ExtractNextDecodableTemporalUnit(),
              ElementsAre(FrameWithId(2)));
  EXPECT_THAT(buffer.ExtractNextDecodableTemporalUnit(),
              ElementsAre(FrameWithId(3)));
  EXPECT_THAT(buffer.ExtractNextDecodableTemporalUnit(),
              ElementsAre(FrameWithId(4)));
  EXPECT_THAT(buffer.ExtractNextDecodableTemporalUnit(),
              ElementsAre(FrameWithId(5)));

  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(70).Id(7).Refs({5}).AsLast().Build()));
  EXPECT_THAT(buffer.ExtractNextDecodableTemporalUnit(),
              ElementsAre(FrameWithId(7)));
  EXPECT_FALSE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(60).Id(6).Refs({4}).AsLast().Build()));
  EXPECT_THAT(buffer.ExtractNextDecodableTemporalUnit(), IsEmpty());
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(90).Id(9).Refs({7}).AsLast().Build()));
  EXPECT_THAT(buffer.ExtractNextDecodableTemporalUnit(),
              ElementsAre(FrameWithId(9)));
}

TEST(FrameBuffer3Test, LegacyFrameIdJumpBehavior) {
  {
    test::ScopedKeyValueConfig field_trials(
        "WebRTC-LegacyFrameIdJumpBehavior/Disabled/");
    FrameBuffer buffer(/*max_frame_slots=*/10, /*max_decode_history=*/100,
                       field_trials);

    EXPECT_TRUE(buffer.InsertFrame(
        test::FakeFrameBuilder().Time(20).Id(3).AsLast().Build()));
    EXPECT_THAT(buffer.ExtractNextDecodableTemporalUnit(),
                ElementsAre(FrameWithId(3)));
    EXPECT_FALSE(buffer.InsertFrame(
        test::FakeFrameBuilder().Time(30).Id(2).AsLast().Build()));
    EXPECT_THAT(buffer.ExtractNextDecodableTemporalUnit(), IsEmpty());
  }

  {
    // WebRTC-LegacyFrameIdJumpBehavior is disabled by default.
    test::ScopedKeyValueConfig field_trials;
    FrameBuffer buffer(/*max_frame_slots=*/10, /*max_decode_history=*/100,
                       field_trials);

    EXPECT_TRUE(buffer.InsertFrame(
        test::FakeFrameBuilder().Time(20).Id(3).AsLast().Build()));
    EXPECT_THAT(buffer.ExtractNextDecodableTemporalUnit(),
                ElementsAre(FrameWithId(3)));
    EXPECT_FALSE(buffer.InsertFrame(
        test::FakeFrameBuilder().Time(30).Id(2).Refs({1}).AsLast().Build()));
    EXPECT_THAT(buffer.ExtractNextDecodableTemporalUnit(), IsEmpty());
    EXPECT_TRUE(buffer.InsertFrame(
        test::FakeFrameBuilder().Time(40).Id(1).AsLast().Build()));
    EXPECT_THAT(buffer.ExtractNextDecodableTemporalUnit(),
                ElementsAre(FrameWithId(1)));
  }
}

TEST(FrameBuffer3Test, TotalNumberOfContinuousTemporalUnits) {
  test::ScopedKeyValueConfig field_trials;
  FrameBuffer buffer(/*max_frame_slots=*/10, /*max_decode_history=*/100,
                     field_trials);
  EXPECT_THAT(buffer.GetTotalNumberOfContinuousTemporalUnits(), Eq(0));

  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(10).Id(1).AsLast().Build()));
  EXPECT_THAT(buffer.GetTotalNumberOfContinuousTemporalUnits(), Eq(1));

  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(20).Id(2).Refs({1}).Build()));
  EXPECT_THAT(buffer.GetTotalNumberOfContinuousTemporalUnits(), Eq(1));

  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(40).Id(4).Refs({2}).Build()));
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(40).Id(5).Refs({3, 4}).AsLast().Build()));
  EXPECT_THAT(buffer.GetTotalNumberOfContinuousTemporalUnits(), Eq(1));

  // Reordered
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(20).Id(3).Refs({2}).AsLast().Build()));
  EXPECT_THAT(buffer.GetTotalNumberOfContinuousTemporalUnits(), Eq(3));
}

TEST(FrameBuffer3Test, TotalNumberOfDroppedFrames) {
  test::ScopedKeyValueConfig field_trials;
  FrameBuffer buffer(/*max_frame_slots=*/10, /*max_decode_history=*/100,
                     field_trials);
  EXPECT_THAT(buffer.GetTotalNumberOfDroppedFrames(), Eq(0));

  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(10).Id(1).AsLast().Build()));
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(20).Id(2).Refs({1}).Build()));
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(20).Id(3).Refs({2}).AsLast().Build()));
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(40).Id(4).Refs({1}).Build()));
  EXPECT_TRUE(buffer.InsertFrame(
      test::FakeFrameBuilder().Time(40).Id(5).Refs({4}).AsLast().Build()));

  buffer.ExtractNextDecodableTemporalUnit();
  EXPECT_THAT(buffer.GetTotalNumberOfDroppedFrames(), Eq(0));

  buffer.DropNextDecodableTemporalUnit();
  EXPECT_THAT(buffer.GetTotalNumberOfDroppedFrames(), Eq(2));

  buffer.ExtractNextDecodableTemporalUnit();
  EXPECT_THAT(buffer.GetTotalNumberOfDroppedFrames(), Eq(2));
}

}  // namespace
}  // namespace webrtc
