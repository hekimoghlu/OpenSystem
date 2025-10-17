/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 15, 2025.
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
#include "pc/jitter_buffer_delay.h"

#include "test/gtest.h"

namespace webrtc {

class JitterBufferDelayTest : public ::testing::Test {
 public:
  JitterBufferDelayTest() {}

 protected:
  JitterBufferDelay delay_;
};

TEST_F(JitterBufferDelayTest, Set) {
  // Delay in seconds.
  delay_.Set(3.0);
  EXPECT_EQ(delay_.GetMs(), 3000);
}

TEST_F(JitterBufferDelayTest, DefaultValue) {
  EXPECT_EQ(delay_.GetMs(), 0);  // Default value is 0ms.
}

TEST_F(JitterBufferDelayTest, Clamping) {
  // In current Jitter Buffer implementation (Audio or Video) maximum supported
  // value is 10000 milliseconds.
  delay_.Set(10.5);
  EXPECT_EQ(delay_.GetMs(), 10000);

  // Test int overflow.
  delay_.Set(21474836470.0);
  EXPECT_EQ(delay_.GetMs(), 10000);

  delay_.Set(-21474836470.0);
  EXPECT_EQ(delay_.GetMs(), 0);

  // Boundary value in seconds to milliseconds conversion.
  delay_.Set(0.0009);
  EXPECT_EQ(delay_.GetMs(), 0);

  delay_.Set(-2.0);
  EXPECT_EQ(delay_.GetMs(), 0);
}

}  // namespace webrtc
