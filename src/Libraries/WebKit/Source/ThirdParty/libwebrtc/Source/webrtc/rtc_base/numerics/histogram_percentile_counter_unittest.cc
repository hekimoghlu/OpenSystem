/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 29, 2024.
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
#include "rtc_base/numerics/histogram_percentile_counter.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "test/gtest.h"

TEST(HistogramPercentileCounterTest, ReturnsCorrectPercentiles) {
  rtc::HistogramPercentileCounter counter(10);
  const std::vector<int> kTestValues = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20};

  EXPECT_FALSE(counter.GetPercentile(0.5f));
  // Pairs of {fraction, percentile value} computed by hand
  // for `kTestValues`.
  const std::vector<std::pair<float, uint32_t>> kTestPercentiles = {
      {0.0f, 1},   {0.01f, 1},  {0.5f, 10}, {0.9f, 18},
      {0.95f, 19}, {0.99f, 20}, {1.0f, 20}};
  for (int value : kTestValues) {
    counter.Add(value);
  }
  for (const auto& test_percentile : kTestPercentiles) {
    EXPECT_EQ(test_percentile.second,
              counter.GetPercentile(test_percentile.first).value_or(0));
  }
}

TEST(HistogramPercentileCounterTest, HandlesEmptySequence) {
  rtc::HistogramPercentileCounter counter(10);
  EXPECT_FALSE(counter.GetPercentile(0.5f));
  counter.Add(1u);
  EXPECT_TRUE(counter.GetPercentile(0.5f));
}
