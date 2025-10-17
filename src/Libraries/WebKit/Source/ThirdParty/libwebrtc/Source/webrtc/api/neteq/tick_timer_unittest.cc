/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 18, 2025.
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
#include "api/neteq/tick_timer.h"

#include <cstdint>
#include <memory>

#include "test/gtest.h"

namespace webrtc {

// Verify that the default value for ms_per_tick is 10.
TEST(TickTimer, DefaultMsPerTick) {
  TickTimer tt;
  EXPECT_EQ(10, tt.ms_per_tick());
}

TEST(TickTimer, CustomMsPerTick) {
  TickTimer tt(17);
  EXPECT_EQ(17, tt.ms_per_tick());
}

TEST(TickTimer, Increment) {
  TickTimer tt;
  EXPECT_EQ(0u, tt.ticks());
  tt.Increment();
  EXPECT_EQ(1u, tt.ticks());

  for (int i = 0; i < 17; ++i) {
    tt.Increment();
  }
  EXPECT_EQ(18u, tt.ticks());

  tt.Increment(17);
  EXPECT_EQ(35u, tt.ticks());
}

TEST(TickTimer, WrapAround) {
  TickTimer tt;
  tt.Increment(UINT64_MAX);
  EXPECT_EQ(UINT64_MAX, tt.ticks());
  tt.Increment();
  EXPECT_EQ(0u, tt.ticks());
}

TEST(TickTimer, Stopwatch) {
  TickTimer tt;
  // Increment it a "random" number of steps.
  tt.Increment(17);

  std::unique_ptr<TickTimer::Stopwatch> sw = tt.GetNewStopwatch();
  ASSERT_TRUE(sw);

  EXPECT_EQ(0u, sw->ElapsedTicks());  // Starts at zero.
  EXPECT_EQ(0u, sw->ElapsedMs());
  tt.Increment();
  EXPECT_EQ(1u, sw->ElapsedTicks());  // Increases with the TickTimer.
  EXPECT_EQ(10u, sw->ElapsedMs());
}

TEST(TickTimer, StopwatchWrapAround) {
  TickTimer tt;
  tt.Increment(UINT64_MAX);

  std::unique_ptr<TickTimer::Stopwatch> sw = tt.GetNewStopwatch();
  ASSERT_TRUE(sw);

  tt.Increment();
  EXPECT_EQ(0u, tt.ticks());
  EXPECT_EQ(1u, sw->ElapsedTicks());
  EXPECT_EQ(10u, sw->ElapsedMs());

  tt.Increment();
  EXPECT_EQ(1u, tt.ticks());
  EXPECT_EQ(2u, sw->ElapsedTicks());
  EXPECT_EQ(20u, sw->ElapsedMs());
}

TEST(TickTimer, StopwatchMsOverflow) {
  TickTimer tt;
  std::unique_ptr<TickTimer::Stopwatch> sw = tt.GetNewStopwatch();
  ASSERT_TRUE(sw);

  tt.Increment(UINT64_MAX / 10);
  EXPECT_EQ(UINT64_MAX, sw->ElapsedMs());

  tt.Increment();
  EXPECT_EQ(UINT64_MAX, sw->ElapsedMs());

  tt.Increment(UINT64_MAX - tt.ticks());
  EXPECT_EQ(UINT64_MAX, tt.ticks());
  EXPECT_EQ(UINT64_MAX, sw->ElapsedMs());
}

TEST(TickTimer, StopwatchWithCustomTicktime) {
  const int kMsPerTick = 17;
  TickTimer tt(kMsPerTick);
  std::unique_ptr<TickTimer::Stopwatch> sw = tt.GetNewStopwatch();
  ASSERT_TRUE(sw);

  EXPECT_EQ(0u, sw->ElapsedMs());
  tt.Increment();
  EXPECT_EQ(static_cast<uint64_t>(kMsPerTick), sw->ElapsedMs());
}

TEST(TickTimer, Countdown) {
  TickTimer tt;
  // Increment it a "random" number of steps.
  tt.Increment(4711);

  std::unique_ptr<TickTimer::Countdown> cd = tt.GetNewCountdown(17);
  ASSERT_TRUE(cd);

  EXPECT_FALSE(cd->Finished());
  tt.Increment();
  EXPECT_FALSE(cd->Finished());

  tt.Increment(16);  // Total increment is now 17.
  EXPECT_TRUE(cd->Finished());

  // Further increments do not change the state.
  tt.Increment();
  EXPECT_TRUE(cd->Finished());
  tt.Increment(1234);
  EXPECT_TRUE(cd->Finished());
}
}  // namespace webrtc
