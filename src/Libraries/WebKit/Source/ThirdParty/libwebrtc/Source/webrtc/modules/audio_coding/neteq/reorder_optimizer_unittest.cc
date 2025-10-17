/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 15, 2021.
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
#include "modules/audio_coding/neteq/reorder_optimizer.h"

#include "test/gtest.h"

namespace webrtc {

namespace {

constexpr int kForgetFactor = 32745;  // 0.9993 in Q15.
constexpr int kMsPerLossPercent = 20;
constexpr int kStartForgetWeight = 1;

}  // namespace

TEST(ReorderOptimizerTest, OnlyIncreaseDelayForReorderedPackets) {
  ReorderOptimizer reorder_optimizer(kForgetFactor, kMsPerLossPercent,
                                     kStartForgetWeight);
  EXPECT_FALSE(reorder_optimizer.GetOptimalDelayMs());

  // Delay should not increase for in-order packets.
  reorder_optimizer.Update(60, /*reordered=*/false, 0);
  EXPECT_EQ(reorder_optimizer.GetOptimalDelayMs(), 20);

  reorder_optimizer.Update(100, /*reordered=*/false, 0);
  EXPECT_EQ(reorder_optimizer.GetOptimalDelayMs(), 20);

  reorder_optimizer.Update(80, /*reordered=*/true, 0);
  EXPECT_EQ(reorder_optimizer.GetOptimalDelayMs(), 100);
}

TEST(ReorderOptimizerTest, AvoidIncreasingDelayWhenProbabilityIsLow) {
  ReorderOptimizer reorder_optimizer(kForgetFactor, kMsPerLossPercent,
                                     kStartForgetWeight);

  reorder_optimizer.Update(40, /*reordered=*/true, 0);
  reorder_optimizer.Update(40, /*reordered=*/true, 0);
  reorder_optimizer.Update(40, /*reordered=*/true, 0);
  EXPECT_EQ(reorder_optimizer.GetOptimalDelayMs(), 60);

  // The cost of the delay is too high relative the probability.
  reorder_optimizer.Update(600, /*reordered=*/true, 0);
  EXPECT_EQ(reorder_optimizer.GetOptimalDelayMs(), 60);
}

TEST(ReorderOptimizerTest, BaseDelayIsSubtractedFromCost) {
  constexpr int kBaseDelayMs = 200;
  ReorderOptimizer reorder_optimizer(kForgetFactor, kMsPerLossPercent,
                                     kStartForgetWeight);

  reorder_optimizer.Update(40, /*reordered=*/true, kBaseDelayMs);
  reorder_optimizer.Update(40, /*reordered=*/true, kBaseDelayMs);
  reorder_optimizer.Update(40, /*reordered=*/true, kBaseDelayMs);
  EXPECT_EQ(reorder_optimizer.GetOptimalDelayMs(), 60);

  // The cost of the delay is too high relative the probability.
  reorder_optimizer.Update(600, /*reordered=*/true, kBaseDelayMs);
  EXPECT_EQ(reorder_optimizer.GetOptimalDelayMs(), 620);
}

}  // namespace webrtc
