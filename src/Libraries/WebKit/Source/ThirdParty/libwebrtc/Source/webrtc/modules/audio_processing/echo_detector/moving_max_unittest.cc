/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 24, 2025.
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
#include "modules/audio_processing/echo_detector/moving_max.h"

#include "test/gtest.h"

namespace webrtc {

// Test if the maximum is correctly found.
TEST(MovingMaxTests, SimpleTest) {
  MovingMax test_moving_max(5);
  test_moving_max.Update(1.0f);
  test_moving_max.Update(1.1f);
  test_moving_max.Update(1.9f);
  test_moving_max.Update(1.87f);
  test_moving_max.Update(1.89f);
  EXPECT_EQ(1.9f, test_moving_max.max());
}

// Test if values fall out of the window when expected.
TEST(MovingMaxTests, SlidingWindowTest) {
  MovingMax test_moving_max(5);
  test_moving_max.Update(1.0f);
  test_moving_max.Update(1.9f);
  test_moving_max.Update(1.7f);
  test_moving_max.Update(1.87f);
  test_moving_max.Update(1.89f);
  test_moving_max.Update(1.3f);
  test_moving_max.Update(1.2f);
  EXPECT_LT(test_moving_max.max(), 1.9f);
}

// Test if Clear() works as expected.
TEST(MovingMaxTests, ClearTest) {
  MovingMax test_moving_max(5);
  test_moving_max.Update(1.0f);
  test_moving_max.Update(1.1f);
  test_moving_max.Update(1.9f);
  test_moving_max.Update(1.87f);
  test_moving_max.Update(1.89f);
  EXPECT_EQ(1.9f, test_moving_max.max());
  test_moving_max.Clear();
  EXPECT_EQ(0.f, test_moving_max.max());
}

// Test the decay of the estimated maximum.
TEST(MovingMaxTests, DecayTest) {
  MovingMax test_moving_max(1);
  test_moving_max.Update(1.0f);
  float previous_value = 1.0f;
  for (int i = 0; i < 500; i++) {
    test_moving_max.Update(0.0f);
    EXPECT_LT(test_moving_max.max(), previous_value);
    EXPECT_GT(test_moving_max.max(), 0.0f);
    previous_value = test_moving_max.max();
  }
  EXPECT_LT(test_moving_max.max(), 0.01f);
}

}  // namespace webrtc
