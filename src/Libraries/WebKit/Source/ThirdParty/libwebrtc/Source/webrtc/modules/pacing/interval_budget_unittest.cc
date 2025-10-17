/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 17, 2021.
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
#include "modules/pacing/interval_budget.h"

#include <cstddef>

#include "test/gtest.h"

namespace webrtc {

namespace {
constexpr int kWindowMs = 500;
constexpr int kBitrateKbps = 100;
constexpr bool kCanBuildUpUnderuse = true;
constexpr bool kCanNotBuildUpUnderuse = false;
size_t TimeToBytes(int bitrate_kbps, int time_ms) {
  return static_cast<size_t>(bitrate_kbps * time_ms / 8);
}
}  // namespace

TEST(IntervalBudgetTest, InitailState) {
  IntervalBudget interval_budget(kBitrateKbps);
  EXPECT_DOUBLE_EQ(interval_budget.budget_ratio(), 0.0);
  EXPECT_EQ(interval_budget.bytes_remaining(), 0u);
}

TEST(IntervalBudgetTest, Underuse) {
  IntervalBudget interval_budget(kBitrateKbps);
  int delta_time_ms = 50;
  interval_budget.IncreaseBudget(delta_time_ms);
  EXPECT_DOUBLE_EQ(interval_budget.budget_ratio(),
                   kWindowMs / static_cast<double>(100 * delta_time_ms));
  EXPECT_EQ(interval_budget.bytes_remaining(),
            TimeToBytes(kBitrateKbps, delta_time_ms));
}

TEST(IntervalBudgetTest, DontUnderuseMoreThanMaxWindow) {
  IntervalBudget interval_budget(kBitrateKbps);
  int delta_time_ms = 1000;
  interval_budget.IncreaseBudget(delta_time_ms);
  EXPECT_DOUBLE_EQ(interval_budget.budget_ratio(), 1.0);
  EXPECT_EQ(interval_budget.bytes_remaining(),
            TimeToBytes(kBitrateKbps, kWindowMs));
}

TEST(IntervalBudgetTest, DontUnderuseMoreThanMaxWindowWhenChangeBitrate) {
  IntervalBudget interval_budget(kBitrateKbps);
  int delta_time_ms = kWindowMs / 2;
  interval_budget.IncreaseBudget(delta_time_ms);
  interval_budget.set_target_rate_kbps(kBitrateKbps / 10);
  EXPECT_DOUBLE_EQ(interval_budget.budget_ratio(), 1.0);
  EXPECT_EQ(interval_budget.bytes_remaining(),
            TimeToBytes(kBitrateKbps / 10, kWindowMs));
}

TEST(IntervalBudgetTest, BalanceChangeOnBitrateChange) {
  IntervalBudget interval_budget(kBitrateKbps);
  int delta_time_ms = kWindowMs;
  interval_budget.IncreaseBudget(delta_time_ms);
  interval_budget.set_target_rate_kbps(kBitrateKbps * 2);
  EXPECT_DOUBLE_EQ(interval_budget.budget_ratio(), 0.5);
  EXPECT_EQ(interval_budget.bytes_remaining(),
            TimeToBytes(kBitrateKbps, kWindowMs));
}

TEST(IntervalBudgetTest, Overuse) {
  IntervalBudget interval_budget(kBitrateKbps);
  int overuse_time_ms = 50;
  int used_bytes = TimeToBytes(kBitrateKbps, overuse_time_ms);
  interval_budget.UseBudget(used_bytes);
  EXPECT_DOUBLE_EQ(interval_budget.budget_ratio(),
                   -kWindowMs / static_cast<double>(100 * overuse_time_ms));
  EXPECT_EQ(interval_budget.bytes_remaining(), 0u);
}

TEST(IntervalBudgetTest, DontOveruseMoreThanMaxWindow) {
  IntervalBudget interval_budget(kBitrateKbps);
  int overuse_time_ms = 1000;
  int used_bytes = TimeToBytes(kBitrateKbps, overuse_time_ms);
  interval_budget.UseBudget(used_bytes);
  EXPECT_DOUBLE_EQ(interval_budget.budget_ratio(), -1.0);
  EXPECT_EQ(interval_budget.bytes_remaining(), 0u);
}

TEST(IntervalBudgetTest, CanBuildUpUnderuseWhenConfigured) {
  IntervalBudget interval_budget(kBitrateKbps, kCanBuildUpUnderuse);
  int delta_time_ms = 50;
  interval_budget.IncreaseBudget(delta_time_ms);
  EXPECT_DOUBLE_EQ(interval_budget.budget_ratio(),
                   kWindowMs / static_cast<double>(100 * delta_time_ms));
  EXPECT_EQ(interval_budget.bytes_remaining(),
            TimeToBytes(kBitrateKbps, delta_time_ms));

  interval_budget.IncreaseBudget(delta_time_ms);
  EXPECT_DOUBLE_EQ(interval_budget.budget_ratio(),
                   2 * kWindowMs / static_cast<double>(100 * delta_time_ms));
  EXPECT_EQ(interval_budget.bytes_remaining(),
            TimeToBytes(kBitrateKbps, 2 * delta_time_ms));
}

TEST(IntervalBudgetTest, CanNotBuildUpUnderuseWhenConfigured) {
  IntervalBudget interval_budget(kBitrateKbps, kCanNotBuildUpUnderuse);
  int delta_time_ms = 50;
  interval_budget.IncreaseBudget(delta_time_ms);
  EXPECT_DOUBLE_EQ(interval_budget.budget_ratio(),
                   kWindowMs / static_cast<double>(100 * delta_time_ms));
  EXPECT_EQ(interval_budget.bytes_remaining(),
            TimeToBytes(kBitrateKbps, delta_time_ms));

  interval_budget.IncreaseBudget(delta_time_ms);
  EXPECT_DOUBLE_EQ(interval_budget.budget_ratio(),
                   kWindowMs / static_cast<double>(100 * delta_time_ms));
  EXPECT_EQ(interval_budget.bytes_remaining(),
            TimeToBytes(kBitrateKbps, delta_time_ms));
}

}  // namespace webrtc
