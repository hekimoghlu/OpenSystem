/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 25, 2022.
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
#include "rtc_base/fake_clock.h"

#include "test/gtest.h"

namespace rtc {
TEST(ScopedFakeClockTest, OverridesGlobalClock) {
  const int64_t kFixedTimeUs = 100000;
  int64_t real_time_us = rtc::TimeMicros();
  EXPECT_NE(real_time_us, 0);
  {
    ScopedFakeClock scoped;
    EXPECT_EQ(rtc::TimeMicros(), 0);

    scoped.AdvanceTime(webrtc::TimeDelta::Millis(1));
    EXPECT_EQ(rtc::TimeMicros(), 1000);

    scoped.SetTime(webrtc::Timestamp::Micros(kFixedTimeUs));
    EXPECT_EQ(rtc::TimeMicros(), kFixedTimeUs);

    scoped.AdvanceTime(webrtc::TimeDelta::Millis(1));
    EXPECT_EQ(rtc::TimeMicros(), kFixedTimeUs + 1000);
  }

  EXPECT_NE(rtc::TimeMicros(), kFixedTimeUs + 1000);
  EXPECT_GE(rtc::TimeMicros(), real_time_us);
}
}  // namespace rtc
