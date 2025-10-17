/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 14, 2022.
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
#include "modules/audio_processing/echo_detector/circular_buffer.h"

#include "test/gtest.h"

namespace webrtc {

TEST(CircularBufferTests, LessThanMaxTest) {
  CircularBuffer test_buffer(3);
  test_buffer.Push(1.f);
  test_buffer.Push(2.f);
  EXPECT_EQ(1.f, test_buffer.Pop());
  EXPECT_EQ(2.f, test_buffer.Pop());
}

TEST(CircularBufferTests, FillTest) {
  CircularBuffer test_buffer(3);
  test_buffer.Push(1.f);
  test_buffer.Push(2.f);
  test_buffer.Push(3.f);
  EXPECT_EQ(1.f, test_buffer.Pop());
  EXPECT_EQ(2.f, test_buffer.Pop());
  EXPECT_EQ(3.f, test_buffer.Pop());
}

TEST(CircularBufferTests, OverflowTest) {
  CircularBuffer test_buffer(3);
  test_buffer.Push(1.f);
  test_buffer.Push(2.f);
  test_buffer.Push(3.f);
  test_buffer.Push(4.f);
  // Because the circular buffer has a size of 3, the first insert should have
  // been forgotten.
  EXPECT_EQ(2.f, test_buffer.Pop());
  EXPECT_EQ(3.f, test_buffer.Pop());
  EXPECT_EQ(4.f, test_buffer.Pop());
}

TEST(CircularBufferTests, ReadFromEmpty) {
  CircularBuffer test_buffer(3);
  EXPECT_EQ(std::nullopt, test_buffer.Pop());
}

}  // namespace webrtc
