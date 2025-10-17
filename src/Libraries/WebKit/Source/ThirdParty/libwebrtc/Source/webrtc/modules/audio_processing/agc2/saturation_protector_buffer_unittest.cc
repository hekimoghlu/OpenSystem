/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 25, 2023.
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
#include "modules/audio_processing/agc2/saturation_protector_buffer.h"

#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

using ::testing::Eq;
using ::testing::Optional;

TEST(GainController2SaturationProtectorBuffer, Init) {
  SaturationProtectorBuffer b;
  EXPECT_EQ(b.Size(), 0);
  EXPECT_FALSE(b.Front().has_value());
}

TEST(GainController2SaturationProtectorBuffer, PushBack) {
  SaturationProtectorBuffer b;
  constexpr float kValue = 123.0f;
  b.PushBack(kValue);
  EXPECT_EQ(b.Size(), 1);
  EXPECT_THAT(b.Front(), Optional(Eq(kValue)));
}

TEST(GainController2SaturationProtectorBuffer, Reset) {
  SaturationProtectorBuffer b;
  b.PushBack(123.0f);
  b.Reset();
  EXPECT_EQ(b.Size(), 0);
  EXPECT_FALSE(b.Front().has_value());
}

// Checks that the front value does not change until the ring buffer gets full.
TEST(GainController2SaturationProtectorBuffer, FrontUntilBufferIsFull) {
  SaturationProtectorBuffer b;
  constexpr float kValue = 123.0f;
  b.PushBack(kValue);
  for (int i = 1; i < b.Capacity(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_THAT(b.Front(), Optional(Eq(kValue)));
    b.PushBack(kValue + i);
  }
}

// Checks that when the buffer is full it behaves as a shift register.
TEST(GainController2SaturationProtectorBuffer, FrontIsDelayed) {
  SaturationProtectorBuffer b;
  // Fill the buffer.
  for (int i = 0; i < b.Capacity(); ++i) {
    b.PushBack(i);
  }
  // The ring buffer should now behave as a shift register with a delay equal to
  // its capacity.
  for (int i = b.Capacity(); i < 2 * b.Capacity() + 1; ++i) {
    SCOPED_TRACE(i);
    EXPECT_THAT(b.Front(), Optional(Eq(i - b.Capacity())));
    b.PushBack(i);
  }
}

}  // namespace
}  // namespace webrtc
