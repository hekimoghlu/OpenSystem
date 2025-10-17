/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 30, 2025.
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
#include "video/report_block_stats.h"

#include "test/gtest.h"

namespace webrtc {
namespace {

constexpr uint32_t kSsrc1 = 123;
constexpr uint32_t kSsrc2 = 234;

TEST(ReportBlockStatsTest, StoreAndGetFractionLost) {
  ReportBlockStats stats;
  EXPECT_EQ(-1, stats.FractionLostInPercent());

  // First report.
  stats.Store(kSsrc1, /*packets_lost=*/10,
              /*extended_highest_sequence_number=*/24'000);
  EXPECT_EQ(-1, stats.FractionLostInPercent());
  // fl: 100 * (15-10) / (24100-24000) = 5%
  stats.Store(kSsrc1, /*packets_lost=*/15,
              /*extended_highest_sequence_number=*/24'100);
  EXPECT_EQ(5, stats.FractionLostInPercent());
  // fl: 100 * (50-10) / (24200-24000) = 20%
  stats.Store(kSsrc1, /*packets_lost=*/50,
              /*extended_highest_sequence_number=*/24'200);
  EXPECT_EQ(20, stats.FractionLostInPercent());
}

TEST(ReportBlockStatsTest, StoreAndGetFractionLost_TwoSsrcs) {
  ReportBlockStats stats;
  EXPECT_EQ(-1, stats.FractionLostInPercent());

  // First report.
  stats.Store(kSsrc1, /*packets_lost=*/10,
              /*extended_highest_sequence_number=*/24'000);
  EXPECT_EQ(-1, stats.FractionLostInPercent());
  // fl: 100 * (15-10) / (24100-24000) = 5%
  stats.Store(kSsrc1, /*packets_lost=*/15,
              /*extended_highest_sequence_number=*/24'100);
  EXPECT_EQ(5, stats.FractionLostInPercent());

  // First report, kSsrc2.
  stats.Store(kSsrc2, /*packets_lost=*/111,
              /*extended_highest_sequence_number=*/8'500);
  EXPECT_EQ(5, stats.FractionLostInPercent());
  // fl: 100 * ((15-10) + (136-111)) / ((24100-24000) + (8800-8500)) = 7%
  stats.Store(kSsrc2, /*packets_lost=*/136,
              /*extended_highest_sequence_number=*/8'800);
  EXPECT_EQ(7, stats.FractionLostInPercent());
}

}  // namespace
}  // namespace webrtc
