/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 21, 2023.
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
#include "modules/video_coding/codecs/test/videocodec_test_stats_impl.h"

#include <vector>

#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {
namespace test {

using FrameStatistics = VideoCodecTestStatsImpl::FrameStatistics;

namespace {

const size_t kTimestamp = 12345;

using ::testing::AllOf;
using ::testing::Contains;
using ::testing::Field;

}  // namespace

TEST(StatsTest, AddAndGetFrame) {
  VideoCodecTestStatsImpl stats;
  stats.AddFrame(FrameStatistics(0, kTimestamp, 0));
  FrameStatistics* frame_stat = stats.GetFrame(0u, 0);
  EXPECT_EQ(0u, frame_stat->frame_number);
  EXPECT_EQ(kTimestamp, frame_stat->rtp_timestamp);
}

TEST(StatsTest, GetOrAddFrame_noFrame_createsNewFrameStat) {
  VideoCodecTestStatsImpl stats;
  stats.GetOrAddFrame(kTimestamp, 0);
  FrameStatistics* frame_stat = stats.GetFrameWithTimestamp(kTimestamp, 0);
  EXPECT_EQ(kTimestamp, frame_stat->rtp_timestamp);
}

TEST(StatsTest, GetOrAddFrame_frameExists_returnsExistingFrameStat) {
  VideoCodecTestStatsImpl stats;
  stats.AddFrame(FrameStatistics(0, kTimestamp, 0));
  FrameStatistics* frame_stat1 = stats.GetFrameWithTimestamp(kTimestamp, 0);
  FrameStatistics* frame_stat2 = stats.GetOrAddFrame(kTimestamp, 0);
  EXPECT_EQ(frame_stat1, frame_stat2);
}

TEST(StatsTest, AddAndGetFrames) {
  VideoCodecTestStatsImpl stats;
  const size_t kNumFrames = 1000;
  for (size_t i = 0; i < kNumFrames; ++i) {
    stats.AddFrame(FrameStatistics(i, kTimestamp + i, 0));
    FrameStatistics* frame_stat = stats.GetFrame(i, 0);
    EXPECT_EQ(i, frame_stat->frame_number);
    EXPECT_EQ(kTimestamp + i, frame_stat->rtp_timestamp);
  }
  EXPECT_EQ(kNumFrames, stats.Size(0));
  // Get frame.
  size_t i = 22;
  FrameStatistics* frame_stat = stats.GetFrameWithTimestamp(kTimestamp + i, 0);
  EXPECT_EQ(i, frame_stat->frame_number);
  EXPECT_EQ(kTimestamp + i, frame_stat->rtp_timestamp);
}

TEST(StatsTest, AddFrameLayering) {
  VideoCodecTestStatsImpl stats;
  for (size_t i = 0; i < 3; ++i) {
    stats.AddFrame(FrameStatistics(0, kTimestamp + i, i));
    FrameStatistics* frame_stat = stats.GetFrame(0u, i);
    EXPECT_EQ(0u, frame_stat->frame_number);
    EXPECT_EQ(kTimestamp, frame_stat->rtp_timestamp - i);
    EXPECT_EQ(1u, stats.Size(i));
  }
}

TEST(StatsTest, GetFrameStatistics) {
  VideoCodecTestStatsImpl stats;

  stats.AddFrame(FrameStatistics(0, kTimestamp, 0));
  stats.AddFrame(FrameStatistics(0, kTimestamp, 1));
  stats.AddFrame(FrameStatistics(1, kTimestamp + 3000, 0));
  stats.AddFrame(FrameStatistics(1, kTimestamp + 3000, 1));

  const std::vector<FrameStatistics> frame_stats = stats.GetFrameStatistics();

  auto field_matcher = [](size_t frame_number, size_t spatial_idx) {
    return AllOf(Field(&FrameStatistics::frame_number, frame_number),
                 Field(&FrameStatistics::spatial_idx, spatial_idx));
  };
  EXPECT_THAT(frame_stats, Contains(field_matcher(0, 0)));
  EXPECT_THAT(frame_stats, Contains(field_matcher(0, 1)));
  EXPECT_THAT(frame_stats, Contains(field_matcher(1, 0)));
  EXPECT_THAT(frame_stats, Contains(field_matcher(1, 1)));
}

}  // namespace test
}  // namespace webrtc
