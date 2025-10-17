/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 13, 2023.
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
#include "modules/audio_processing/aec3/clockdrift_detector.h"

#include "test/gtest.h"

namespace webrtc {
TEST(ClockdriftDetector, ClockdriftDetector) {
  ClockdriftDetector c;
  // No clockdrift at start.
  EXPECT_TRUE(c.ClockdriftLevel() == ClockdriftDetector::Level::kNone);

  // Monotonically increasing delay.
  for (int i = 0; i < 100; i++)
    c.Update(1000);
  EXPECT_TRUE(c.ClockdriftLevel() == ClockdriftDetector::Level::kNone);
  for (int i = 0; i < 100; i++)
    c.Update(1001);
  EXPECT_TRUE(c.ClockdriftLevel() == ClockdriftDetector::Level::kNone);
  for (int i = 0; i < 100; i++)
    c.Update(1002);
  // Probable clockdrift.
  EXPECT_TRUE(c.ClockdriftLevel() == ClockdriftDetector::Level::kProbable);
  for (int i = 0; i < 100; i++)
    c.Update(1003);
  // Verified clockdrift.
  EXPECT_TRUE(c.ClockdriftLevel() == ClockdriftDetector::Level::kVerified);

  // Stable delay.
  for (int i = 0; i < 10000; i++)
    c.Update(1003);
  // No clockdrift.
  EXPECT_TRUE(c.ClockdriftLevel() == ClockdriftDetector::Level::kNone);

  // Decreasing delay.
  for (int i = 0; i < 100; i++)
    c.Update(1001);
  for (int i = 0; i < 100; i++)
    c.Update(999);
  // Probable clockdrift.
  EXPECT_TRUE(c.ClockdriftLevel() == ClockdriftDetector::Level::kProbable);
  for (int i = 0; i < 100; i++)
    c.Update(1000);
  for (int i = 0; i < 100; i++)
    c.Update(998);
  // Verified clockdrift.
  EXPECT_TRUE(c.ClockdriftLevel() == ClockdriftDetector::Level::kVerified);
}
}  // namespace webrtc
