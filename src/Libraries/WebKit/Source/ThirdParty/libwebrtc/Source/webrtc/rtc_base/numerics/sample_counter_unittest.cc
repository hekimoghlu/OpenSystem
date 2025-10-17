/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 3, 2022.
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
#include "rtc_base/numerics/sample_counter.h"

#include <initializer_list>
#include <optional>

#include "test/gmock.h"
#include "test/gtest.h"

using ::testing::Eq;

namespace rtc {

TEST(SampleCounterTest, ProcessesNoSamples) {
  constexpr int kMinSamples = 1;
  SampleCounter counter;
  EXPECT_THAT(counter.Avg(kMinSamples), Eq(std::nullopt));
  EXPECT_THAT(counter.Max(), Eq(std::nullopt));
  EXPECT_THAT(counter.Min(), Eq(std::nullopt));
}

TEST(SampleCounterTest, NotEnoughSamples) {
  constexpr int kMinSamples = 6;
  SampleCounter counter;
  for (int value : {1, 2, 3, 4, 5}) {
    counter.Add(value);
  }
  EXPECT_THAT(counter.Avg(kMinSamples), Eq(std::nullopt));
  EXPECT_THAT(counter.Sum(kMinSamples), Eq(std::nullopt));
  EXPECT_THAT(counter.Max(), Eq(5));
  EXPECT_THAT(counter.Min(), Eq(1));
}

TEST(SampleCounterTest, EnoughSamples) {
  constexpr int kMinSamples = 5;
  SampleCounter counter;
  for (int value : {1, 2, 3, 4, 5}) {
    counter.Add(value);
  }
  EXPECT_THAT(counter.Avg(kMinSamples), Eq(3));
  EXPECT_THAT(counter.Sum(kMinSamples), Eq(15));
  EXPECT_THAT(counter.Max(), Eq(5));
  EXPECT_THAT(counter.Min(), Eq(1));
}

TEST(SampleCounterTest, ComputesVariance) {
  constexpr int kMinSamples = 5;
  SampleCounterWithVariance counter;
  for (int value : {1, 2, 3, 4, 5}) {
    counter.Add(value);
  }
  EXPECT_THAT(counter.Variance(kMinSamples), Eq(2));
}

TEST(SampleCounterTest, AggregatesTwoCounters) {
  constexpr int kMinSamples = 5;
  SampleCounterWithVariance counter1;
  for (int value : {1, 2, 3}) {
    counter1.Add(value);
  }
  SampleCounterWithVariance counter2;
  for (int value : {4, 5}) {
    counter2.Add(value);
  }
  // Before aggregation there is not enough samples.
  EXPECT_THAT(counter1.Avg(kMinSamples), Eq(std::nullopt));
  EXPECT_THAT(counter1.Variance(kMinSamples), Eq(std::nullopt));
  // Aggregate counter2 in counter1.
  counter1.Add(counter2);
  EXPECT_THAT(counter1.Avg(kMinSamples), Eq(3));
  EXPECT_THAT(counter1.Max(), Eq(5));
  EXPECT_THAT(counter1.Min(), Eq(1));
  EXPECT_THAT(counter1.Variance(kMinSamples), Eq(2));
}

}  // namespace rtc
