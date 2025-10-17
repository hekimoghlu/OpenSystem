/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 17, 2021.
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
#include "modules/video_coding/utility/decoded_frames_history.h"

#include "test/gtest.h"

namespace webrtc {
namespace video_coding {
namespace {

constexpr int kHistorySize = 1 << 13;

TEST(DecodedFramesHistory, RequestOnEmptyHistory) {
  DecodedFramesHistory history(kHistorySize);
  EXPECT_EQ(history.WasDecoded(1234), false);
}

TEST(DecodedFramesHistory, FindsLastDecodedFrame) {
  DecodedFramesHistory history(kHistorySize);
  history.InsertDecoded(1234, 0);
  EXPECT_EQ(history.WasDecoded(1234), true);
}

TEST(DecodedFramesHistory, FindsPreviousFrame) {
  DecodedFramesHistory history(kHistorySize);
  history.InsertDecoded(1234, 0);
  history.InsertDecoded(1235, 0);
  EXPECT_EQ(history.WasDecoded(1234), true);
}

TEST(DecodedFramesHistory, ReportsMissingFrame) {
  DecodedFramesHistory history(kHistorySize);
  history.InsertDecoded(1234, 0);
  history.InsertDecoded(1236, 0);
  EXPECT_EQ(history.WasDecoded(1235), false);
}

TEST(DecodedFramesHistory, ClearsHistory) {
  DecodedFramesHistory history(kHistorySize);
  history.InsertDecoded(1234, 0);
  history.Clear();
  EXPECT_EQ(history.WasDecoded(1234), false);
  EXPECT_EQ(history.GetLastDecodedFrameId(), std::nullopt);
  EXPECT_EQ(history.GetLastDecodedFrameTimestamp(), std::nullopt);
}

TEST(DecodedFramesHistory, HandlesBigJumpInPictureId) {
  DecodedFramesHistory history(kHistorySize);
  history.InsertDecoded(1234, 0);
  history.InsertDecoded(1235, 0);
  history.InsertDecoded(1236, 0);
  history.InsertDecoded(1236 + kHistorySize / 2, 0);
  EXPECT_EQ(history.WasDecoded(1234), true);
  EXPECT_EQ(history.WasDecoded(1237), false);
}

TEST(DecodedFramesHistory, ForgetsTooOldHistory) {
  DecodedFramesHistory history(kHistorySize);
  history.InsertDecoded(1234, 0);
  history.InsertDecoded(1235, 0);
  history.InsertDecoded(1236, 0);
  history.InsertDecoded(1236 + kHistorySize * 2, 0);
  EXPECT_EQ(history.WasDecoded(1234), false);
  EXPECT_EQ(history.WasDecoded(1237), false);
}

TEST(DecodedFramesHistory, ReturnsLastDecodedFrameId) {
  DecodedFramesHistory history(kHistorySize);
  EXPECT_EQ(history.GetLastDecodedFrameId(), std::nullopt);
  history.InsertDecoded(1234, 0);
  EXPECT_EQ(history.GetLastDecodedFrameId(), 1234);
  history.InsertDecoded(1235, 0);
  EXPECT_EQ(history.GetLastDecodedFrameId(), 1235);
}

TEST(DecodedFramesHistory, ReturnsLastDecodedFrameTimestamp) {
  DecodedFramesHistory history(kHistorySize);
  EXPECT_EQ(history.GetLastDecodedFrameTimestamp(), std::nullopt);
  history.InsertDecoded(1234, 12345);
  EXPECT_EQ(history.GetLastDecodedFrameTimestamp(), 12345u);
  history.InsertDecoded(1235, 12366);
  EXPECT_EQ(history.GetLastDecodedFrameTimestamp(), 12366u);
}

TEST(DecodedFramesHistory, NegativePictureIds) {
  DecodedFramesHistory history(kHistorySize);
  history.InsertDecoded(-1234, 12345);
  history.InsertDecoded(-1233, 12366);
  EXPECT_EQ(*history.GetLastDecodedFrameId(), -1233);

  history.InsertDecoded(-1, 12377);
  history.InsertDecoded(0, 12388);
  EXPECT_EQ(*history.GetLastDecodedFrameId(), 0);

  history.InsertDecoded(1, 12399);
  EXPECT_EQ(*history.GetLastDecodedFrameId(), 1);

  EXPECT_EQ(history.WasDecoded(-1234), true);
  EXPECT_EQ(history.WasDecoded(-1), true);
  EXPECT_EQ(history.WasDecoded(0), true);
  EXPECT_EQ(history.WasDecoded(1), true);
}

}  // namespace
}  // namespace video_coding
}  // namespace webrtc
