/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 3, 2023.
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
#include "rtc_base/numerics/moving_max_counter.h"

#include "test/gtest.h"

TEST(MovingMaxCounter, ReportsMaximumInTheWindow) {
  rtc::MovingMaxCounter<int> counter(100);
  counter.Add(1, 1);
  EXPECT_EQ(counter.Max(1), 1);
  counter.Add(2, 30);
  EXPECT_EQ(counter.Max(30), 2);
  counter.Add(100, 60);
  EXPECT_EQ(counter.Max(60), 100);
  counter.Add(4, 70);
  EXPECT_EQ(counter.Max(70), 100);
  counter.Add(5, 90);
  EXPECT_EQ(counter.Max(90), 100);
}

TEST(MovingMaxCounter, IgnoresOldElements) {
  rtc::MovingMaxCounter<int> counter(100);
  counter.Add(1, 1);
  counter.Add(2, 30);
  counter.Add(100, 60);
  counter.Add(4, 70);
  counter.Add(5, 90);
  EXPECT_EQ(counter.Max(160), 100);
  // 100 is now out of the window. Next maximum is 5.
  EXPECT_EQ(counter.Max(161), 5);
}

TEST(MovingMaxCounter, HandlesEmptyWindow) {
  rtc::MovingMaxCounter<int> counter(100);
  counter.Add(123, 1);
  EXPECT_TRUE(counter.Max(101).has_value());
  EXPECT_FALSE(counter.Max(102).has_value());
}

TEST(MovingMaxCounter, HandlesSamplesWithEqualTimestamps) {
  rtc::MovingMaxCounter<int> counter(100);
  counter.Add(2, 30);
  EXPECT_EQ(counter.Max(30), 2);
  counter.Add(5, 30);
  EXPECT_EQ(counter.Max(30), 5);
  counter.Add(4, 30);
  EXPECT_EQ(counter.Max(30), 5);
  counter.Add(1, 90);
  EXPECT_EQ(counter.Max(150), 1);
}
