/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 19, 2023.
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
#include "modules/video_coding/timing/rtt_filter.h"

#include "api/units/time_delta.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {

TEST(RttFilterTest, RttIsCapped) {
  RttFilter rtt_filter;
  rtt_filter.Update(TimeDelta::Seconds(500));

  EXPECT_EQ(rtt_filter.Rtt(), TimeDelta::Seconds(3));
}

// If the difference between samples is more than away 2.5 stddev from the mean
// then this is considered a jump. After more than 5 data points at the new
// level, the RTT is reset to the new level.
TEST(RttFilterTest, PositiveJumpDetection) {
  RttFilter rtt_filter;

  rtt_filter.Update(TimeDelta::Millis(200));
  rtt_filter.Update(TimeDelta::Millis(200));
  rtt_filter.Update(TimeDelta::Millis(200));

  // Trigger 5 jumps.
  rtt_filter.Update(TimeDelta::Millis(1400));
  rtt_filter.Update(TimeDelta::Millis(1500));
  rtt_filter.Update(TimeDelta::Millis(1600));
  rtt_filter.Update(TimeDelta::Millis(1600));

  EXPECT_EQ(rtt_filter.Rtt(), TimeDelta::Millis(1600));

  rtt_filter.Update(TimeDelta::Millis(1600));
  EXPECT_EQ(rtt_filter.Rtt(), TimeDelta::Millis(1600));
}

TEST(RttFilterTest, NegativeJumpDetection) {
  RttFilter rtt_filter;

  for (int i = 0; i < 10; ++i)
    rtt_filter.Update(TimeDelta::Millis(1500));

  // Trigger 5 negative data points that jump rtt down.
  rtt_filter.Update(TimeDelta::Millis(200));
  rtt_filter.Update(TimeDelta::Millis(200));
  rtt_filter.Update(TimeDelta::Millis(200));
  rtt_filter.Update(TimeDelta::Millis(200));
  // Before 5 data points at the new level, max RTT is still 1500.
  EXPECT_EQ(rtt_filter.Rtt(), TimeDelta::Millis(1500));

  rtt_filter.Update(TimeDelta::Millis(300));
  EXPECT_EQ(rtt_filter.Rtt(), TimeDelta::Millis(300));
}

TEST(RttFilterTest, JumpsResetByDirectionShift) {
  RttFilter rtt_filter;
  for (int i = 0; i < 10; ++i)
    rtt_filter.Update(TimeDelta::Millis(1500));

  // Trigger 4 negative jumps, then a positive one. This resets the jump
  // detection.
  rtt_filter.Update(TimeDelta::Millis(200));
  rtt_filter.Update(TimeDelta::Millis(200));
  rtt_filter.Update(TimeDelta::Millis(200));
  rtt_filter.Update(TimeDelta::Millis(200));
  rtt_filter.Update(TimeDelta::Millis(2000));
  EXPECT_EQ(rtt_filter.Rtt(), TimeDelta::Millis(2000));

  rtt_filter.Update(TimeDelta::Millis(300));
  EXPECT_EQ(rtt_filter.Rtt(), TimeDelta::Millis(2000));
}

// If the difference between the max and average is more than 3.5 stddevs away
// then a drift is detected, and a short filter is applied to find a new max
// rtt.
TEST(RttFilterTest, DriftDetection) {
  RttFilter rtt_filter;

  // Descend RTT by 30ms and settle at 700ms RTT. A drift is detected after rtt
  // of 700ms is reported around 50 times for these targets.
  constexpr TimeDelta kStartRtt = TimeDelta::Millis(1000);
  constexpr TimeDelta kDriftTarget = TimeDelta::Millis(700);
  constexpr TimeDelta kDelta = TimeDelta::Millis(30);
  for (TimeDelta rtt = kStartRtt; rtt >= kDriftTarget; rtt -= kDelta)
    rtt_filter.Update(rtt);

  EXPECT_EQ(rtt_filter.Rtt(), kStartRtt);

  for (int i = 0; i < 50; ++i)
    rtt_filter.Update(kDriftTarget);
  EXPECT_EQ(rtt_filter.Rtt(), kDriftTarget);
}

}  // namespace webrtc
