/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 13, 2023.
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
#include "rtc_base/numerics/moving_average.h"

#include <optional>

#include "test/gtest.h"

namespace test {

TEST(MovingAverageTest, EmptyAverage) {
  rtc::MovingAverage moving_average(1);
  EXPECT_EQ(0u, moving_average.Size());
  EXPECT_EQ(std::nullopt, moving_average.GetAverageRoundedDown());
}

// Test single value.
TEST(MovingAverageTest, OneElement) {
  rtc::MovingAverage moving_average(1);
  moving_average.AddSample(3);
  EXPECT_EQ(1u, moving_average.Size());
  EXPECT_EQ(3, *moving_average.GetAverageRoundedDown());
}

TEST(MovingAverageTest, GetAverage) {
  rtc::MovingAverage moving_average(1024);
  moving_average.AddSample(1);
  moving_average.AddSample(1);
  moving_average.AddSample(3);
  moving_average.AddSample(3);
  EXPECT_EQ(*moving_average.GetAverageRoundedDown(), 2);
  EXPECT_EQ(*moving_average.GetAverageRoundedToClosest(), 2);
}

TEST(MovingAverageTest, GetAverageRoundedDownRounds) {
  rtc::MovingAverage moving_average(1024);
  moving_average.AddSample(1);
  moving_average.AddSample(2);
  moving_average.AddSample(2);
  moving_average.AddSample(2);
  EXPECT_EQ(*moving_average.GetAverageRoundedDown(), 1);
}

TEST(MovingAverageTest, GetAverageRoundedToClosestRounds) {
  rtc::MovingAverage moving_average(1024);
  moving_average.AddSample(1);
  moving_average.AddSample(2);
  moving_average.AddSample(2);
  moving_average.AddSample(2);
  EXPECT_EQ(*moving_average.GetAverageRoundedToClosest(), 2);
}

TEST(MovingAverageTest, Reset) {
  rtc::MovingAverage moving_average(5);
  moving_average.AddSample(1);
  EXPECT_EQ(1, *moving_average.GetAverageRoundedDown());
  EXPECT_EQ(1, *moving_average.GetAverageRoundedToClosest());

  moving_average.Reset();

  EXPECT_FALSE(moving_average.GetAverageRoundedDown());
  moving_average.AddSample(10);
  EXPECT_EQ(10, *moving_average.GetAverageRoundedDown());
  EXPECT_EQ(10, *moving_average.GetAverageRoundedToClosest());
}

TEST(MovingAverageTest, ManySamples) {
  rtc::MovingAverage moving_average(10);
  for (int i = 1; i < 11; i++) {
    moving_average.AddSample(i);
  }
  EXPECT_EQ(*moving_average.GetAverageRoundedDown(), 5);
  EXPECT_EQ(*moving_average.GetAverageRoundedToClosest(), 6);
  for (int i = 1; i < 2001; i++) {
    moving_average.AddSample(i);
  }
  EXPECT_EQ(*moving_average.GetAverageRoundedDown(), 1995);
  EXPECT_EQ(*moving_average.GetAverageRoundedToClosest(), 1996);
}

}  // namespace test
