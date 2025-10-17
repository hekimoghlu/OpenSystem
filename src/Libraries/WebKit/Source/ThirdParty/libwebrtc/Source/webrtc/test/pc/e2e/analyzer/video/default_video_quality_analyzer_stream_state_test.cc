/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 9, 2024.
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
#include "test/pc/e2e/analyzer/video/default_video_quality_analyzer_stream_state.h"

#include <set>

#include "api/units/timestamp.h"
#include "system_wrappers/include/clock.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

TEST(StreamStateTest, PopFrontAndFrontIndependentForEachPeer) {
  StreamState state(/*sender=*/0,
                    /*receivers=*/std::set<size_t>{1, 2}, Timestamp::Seconds(1),
                    Clock::GetRealTimeClock());
  state.PushBack(/*frame_id=*/1);
  state.PushBack(/*frame_id=*/2);

  EXPECT_EQ(state.Front(/*peer=*/1), 1);
  EXPECT_EQ(state.PopFront(/*peer=*/1), 1);
  EXPECT_EQ(state.Front(/*peer=*/1), 2);
  EXPECT_EQ(state.PopFront(/*peer=*/1), 2);
  EXPECT_EQ(state.Front(/*peer=*/2), 1);
  EXPECT_EQ(state.PopFront(/*peer=*/2), 1);
  EXPECT_EQ(state.Front(/*peer=*/2), 2);
  EXPECT_EQ(state.PopFront(/*peer=*/2), 2);
}

TEST(StreamStateTest, IsEmpty) {
  StreamState state(/*sender=*/0,
                    /*receivers=*/std::set<size_t>{1, 2}, Timestamp::Seconds(1),
                    Clock::GetRealTimeClock());
  state.PushBack(/*frame_id=*/1);

  EXPECT_FALSE(state.IsEmpty(/*peer=*/1));

  state.PopFront(/*peer=*/1);

  EXPECT_TRUE(state.IsEmpty(/*peer=*/1));
}

TEST(StreamStateTest, PopFrontForOnlyOnePeerDontChangeAliveFramesCount) {
  StreamState state(/*sender=*/0,
                    /*receivers=*/std::set<size_t>{1, 2}, Timestamp::Seconds(1),
                    Clock::GetRealTimeClock());
  state.PushBack(/*frame_id=*/1);
  state.PushBack(/*frame_id=*/2);

  EXPECT_EQ(state.GetAliveFramesCount(), 2lu);

  state.PopFront(/*peer=*/1);
  state.PopFront(/*peer=*/1);

  EXPECT_EQ(state.GetAliveFramesCount(), 2lu);
}

TEST(StreamStateTest, PopFrontForAllPeersReducesAliveFramesCount) {
  StreamState state(/*sender=*/0,
                    /*receivers=*/std::set<size_t>{1, 2}, Timestamp::Seconds(1),
                    Clock::GetRealTimeClock());
  state.PushBack(/*frame_id=*/1);
  state.PushBack(/*frame_id=*/2);

  EXPECT_EQ(state.GetAliveFramesCount(), 2lu);

  state.PopFront(/*peer=*/1);
  state.PopFront(/*peer=*/2);

  EXPECT_EQ(state.GetAliveFramesCount(), 1lu);
}

TEST(StreamStateTest, RemovePeerForLastExpectedReceiverUpdatesAliveFrames) {
  StreamState state(/*sender=*/0,
                    /*receivers=*/std::set<size_t>{1, 2}, Timestamp::Seconds(1),
                    Clock::GetRealTimeClock());
  state.PushBack(/*frame_id=*/1);
  state.PushBack(/*frame_id=*/2);

  state.PopFront(/*peer=*/1);

  EXPECT_EQ(state.GetAliveFramesCount(), 2lu);

  state.RemovePeer(/*peer=*/2);

  EXPECT_EQ(state.GetAliveFramesCount(), 1lu);
}

}  // namespace
}  // namespace webrtc
