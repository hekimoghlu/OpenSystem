/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 4, 2022.
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
#include "logging/rtc_event_log/events/rtc_event_field_extraction.h"

#include <cstdint>

#include "rtc_base/random.h"
#include "test/gtest.h"

namespace webrtc {

TEST(UnsignedBitWidthTest, SmallValues) {
  EXPECT_EQ(webrtc_event_logging::UnsignedBitWidth(0), 1u);
  EXPECT_EQ(webrtc_event_logging::UnsignedBitWidth(1), 1u);
  EXPECT_EQ(webrtc_event_logging::UnsignedBitWidth(2), 2u);
  EXPECT_EQ(webrtc_event_logging::UnsignedBitWidth(3), 2u);
  EXPECT_EQ(webrtc_event_logging::UnsignedBitWidth(4), 3u);
  EXPECT_EQ(webrtc_event_logging::UnsignedBitWidth(5), 3u);
  EXPECT_EQ(webrtc_event_logging::UnsignedBitWidth(6), 3u);
  EXPECT_EQ(webrtc_event_logging::UnsignedBitWidth(7), 3u);
}

TEST(UnsignedBitWidthTest, PowersOfTwo) {
  EXPECT_EQ(webrtc_event_logging::UnsignedBitWidth(0), 1u);

  for (unsigned i = 0; i < 64; i++) {
    uint64_t x = 1;
    x = x << i;
    EXPECT_EQ(webrtc_event_logging::UnsignedBitWidth(x), i + 1);
  }
}

TEST(UnsignedBitWidthTest, PowersOfTwoMinusOne) {
  for (unsigned i = 1; i < 64; i++) {
    uint64_t x = 1;
    x = (x << i) - 1;
    EXPECT_EQ(webrtc_event_logging::UnsignedBitWidth(x), i);
  }

  uint64_t x = ~static_cast<uint64_t>(0);
  EXPECT_EQ(webrtc_event_logging::UnsignedBitWidth(x), 64u);
}

TEST(UnsignedBitWidthTest, RandomInputs) {
  Random rand(12345);

  for (unsigned i = 0; i < 64; i++) {
    uint64_t x = 1;
    x = x << i;
    uint64_t high = rand.Rand<uint32_t>();
    uint64_t low = rand.Rand<uint32_t>();
    x += ((high << 32) + low) % x;
    EXPECT_EQ(webrtc_event_logging::UnsignedBitWidth(x), i + 1);
  }
}

TEST(SignedBitWidthTest, SignedBitWidth) {
  EXPECT_EQ(webrtc_event_logging::SignedBitWidth(0, 1), 1u);
  EXPECT_EQ(webrtc_event_logging::SignedBitWidth(1, 0), 2u);
  EXPECT_EQ(webrtc_event_logging::SignedBitWidth(1, 2), 2u);
  EXPECT_EQ(webrtc_event_logging::SignedBitWidth(1, 128), 8u);
  EXPECT_EQ(webrtc_event_logging::SignedBitWidth(127, 1), 8u);
  EXPECT_EQ(webrtc_event_logging::SignedBitWidth(127, 128), 8u);
  EXPECT_EQ(webrtc_event_logging::SignedBitWidth(1, 129), 9u);
  EXPECT_EQ(webrtc_event_logging::SignedBitWidth(128, 1), 9u);
}

TEST(MaxUnsignedValueOfBitWidthTest, MaxUnsignedValueOfBitWidth) {
  EXPECT_EQ(webrtc_event_logging::MaxUnsignedValueOfBitWidth(1), 0x01u);
  EXPECT_EQ(webrtc_event_logging::MaxUnsignedValueOfBitWidth(6), 0x3Fu);
  EXPECT_EQ(webrtc_event_logging::MaxUnsignedValueOfBitWidth(8), 0xFFu);
  EXPECT_EQ(webrtc_event_logging::MaxUnsignedValueOfBitWidth(32), 0xFFFFFFFFu);
}

TEST(EncodeAsUnsignedTest, NegativeValues) {
  // Negative values are converted as if cast to unsigned type of
  // the same bitsize using 2-complement representation.
  int16_t x = -1;
  EXPECT_EQ(EncodeAsUnsigned(x), static_cast<uint64_t>(0xFFFF));
  int64_t y = -1;
  EXPECT_EQ(EncodeAsUnsigned(y), static_cast<uint64_t>(0xFFFFFFFFFFFFFFFFull));
}

TEST(EncodeAsUnsignedTest, PositiveValues) {
  // Postive values are unchanged.
  int16_t x = 42;
  EXPECT_EQ(EncodeAsUnsigned(x), static_cast<uint64_t>(42));
  int64_t y = 42;
  EXPECT_EQ(EncodeAsUnsigned(y), static_cast<uint64_t>(42));
}

}  // namespace webrtc
