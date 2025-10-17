/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 19, 2025.
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
#include "modules/audio_coding/neteq/underrun_optimizer.h"

#include "test/gtest.h"

namespace webrtc {

namespace {

constexpr int kDefaultHistogramQuantile = 1020054733;  // 0.95 in Q30.
constexpr int kForgetFactor = 32745;                   // 0.9993 in Q15.

}  // namespace

TEST(UnderrunOptimizerTest, ResamplePacketDelays) {
  TickTimer tick_timer;
  constexpr int kResampleIntervalMs = 500;
  UnderrunOptimizer underrun_optimizer(&tick_timer, kDefaultHistogramQuantile,
                                       kForgetFactor, std::nullopt,
                                       kResampleIntervalMs);

  // The histogram should be updated once with the maximum delay observed for
  // the following sequence of updates.
  for (int i = 0; i < 500; i += 20) {
    underrun_optimizer.Update(i);
    EXPECT_FALSE(underrun_optimizer.GetOptimalDelayMs());
  }
  tick_timer.Increment(kResampleIntervalMs / tick_timer.ms_per_tick() + 1);
  underrun_optimizer.Update(0);
  EXPECT_EQ(underrun_optimizer.GetOptimalDelayMs(), 500);
}

}  // namespace webrtc
