/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 22, 2023.
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
#include "rtc_base/numerics/moving_percentile_filter.h"

#include <stdint.h>

#include <algorithm>
#include <cstddef>

#include "test/gtest.h"

namespace webrtc {

// 25th percentile can be exactly found with a window of length 4.
TEST(MovingPercentileFilter, Percentile25ReturnsMovingPercentile25WithWindow4) {
  MovingPercentileFilter<int> perc25(0.25f, 4);
  const int64_t kSamples[10] = {1, 2, 3, 4, 4, 4, 5, 6, 7, 8};
  const int64_t kExpectedFilteredValues[10] = {1, 1, 1, 1, 2, 3, 4, 4, 4, 5};
  for (size_t i = 0; i < 10; ++i) {
    perc25.Insert(kSamples[i]);
    EXPECT_EQ(kExpectedFilteredValues[i], perc25.GetFilteredValue());
    EXPECT_EQ(std::min<size_t>(i + 1, 4), perc25.GetNumberOfSamplesStored());
  }
}

// 90th percentile becomes the 67th percentile with a window of length 4.
TEST(MovingPercentileFilter, Percentile90ReturnsMovingPercentile67WithWindow4) {
  MovingPercentileFilter<int> perc67(0.67f, 4);
  MovingPercentileFilter<int> perc90(0.9f, 4);
  const int64_t kSamples[8] = {1, 10, 1, 9, 1, 10, 1, 8};
  const int64_t kExpectedFilteredValues[9] = {1, 1, 1, 9, 9, 9, 9, 8};
  for (size_t i = 0; i < 8; ++i) {
    perc67.Insert(kSamples[i]);
    perc90.Insert(kSamples[i]);
    EXPECT_EQ(kExpectedFilteredValues[i], perc67.GetFilteredValue());
    EXPECT_EQ(kExpectedFilteredValues[i], perc90.GetFilteredValue());
  }
}

TEST(MovingMedianFilterTest, ProcessesNoSamples) {
  MovingMedianFilter<int> filter(2);
  EXPECT_EQ(0, filter.GetFilteredValue());
  EXPECT_EQ(0u, filter.GetNumberOfSamplesStored());
}

TEST(MovingMedianFilterTest, ReturnsMovingMedianWindow5) {
  MovingMedianFilter<int> filter(5);
  const int64_t kSamples[5] = {1, 5, 2, 3, 4};
  const int64_t kExpectedFilteredValues[5] = {1, 1, 2, 2, 3};
  for (size_t i = 0; i < 5; ++i) {
    filter.Insert(kSamples[i]);
    EXPECT_EQ(kExpectedFilteredValues[i], filter.GetFilteredValue());
    EXPECT_EQ(i + 1, filter.GetNumberOfSamplesStored());
  }
}

TEST(MovingMedianFilterTest, ReturnsMovingMedianWindow3) {
  MovingMedianFilter<int> filter(3);
  const int64_t kSamples[5] = {1, 5, 2, 3, 4};
  const int64_t kExpectedFilteredValues[5] = {1, 1, 2, 3, 3};
  for (int i = 0; i < 5; ++i) {
    filter.Insert(kSamples[i]);
    EXPECT_EQ(kExpectedFilteredValues[i], filter.GetFilteredValue());
    EXPECT_EQ(std::min<size_t>(i + 1, 3), filter.GetNumberOfSamplesStored());
  }
}

TEST(MovingMedianFilterTest, ReturnsMovingMedianWindow1) {
  MovingMedianFilter<int> filter(1);
  const int64_t kSamples[5] = {1, 5, 2, 3, 4};
  const int64_t kExpectedFilteredValues[5] = {1, 5, 2, 3, 4};
  for (int i = 0; i < 5; ++i) {
    filter.Insert(kSamples[i]);
    EXPECT_EQ(kExpectedFilteredValues[i], filter.GetFilteredValue());
    EXPECT_EQ(1u, filter.GetNumberOfSamplesStored());
  }
}

}  // namespace webrtc
