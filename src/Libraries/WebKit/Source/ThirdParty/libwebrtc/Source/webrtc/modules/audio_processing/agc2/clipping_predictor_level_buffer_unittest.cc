/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 3, 2023.
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
#include "modules/audio_processing/agc2/clipping_predictor_level_buffer.h"

#include <algorithm>

#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

using ::testing::Eq;
using ::testing::Optional;

class ClippingPredictorLevelBufferParametrization
    : public ::testing::TestWithParam<int> {
 protected:
  int capacity() const { return GetParam(); }
};

TEST_P(ClippingPredictorLevelBufferParametrization, CheckEmptyBufferSize) {
  ClippingPredictorLevelBuffer buffer(capacity());
  EXPECT_EQ(buffer.Capacity(), std::max(capacity(), 1));
  EXPECT_EQ(buffer.Size(), 0);
}

TEST_P(ClippingPredictorLevelBufferParametrization, CheckHalfEmptyBufferSize) {
  ClippingPredictorLevelBuffer buffer(capacity());
  for (int i = 0; i < buffer.Capacity() / 2; ++i) {
    buffer.Push({2, 4});
  }
  EXPECT_EQ(buffer.Capacity(), std::max(capacity(), 1));
  EXPECT_EQ(buffer.Size(), std::max(capacity(), 1) / 2);
}

TEST_P(ClippingPredictorLevelBufferParametrization, CheckFullBufferSize) {
  ClippingPredictorLevelBuffer buffer(capacity());
  for (int i = 0; i < buffer.Capacity(); ++i) {
    buffer.Push({2, 4});
  }
  EXPECT_EQ(buffer.Capacity(), std::max(capacity(), 1));
  EXPECT_EQ(buffer.Size(), std::max(capacity(), 1));
}

TEST_P(ClippingPredictorLevelBufferParametrization, CheckLargeBufferSize) {
  ClippingPredictorLevelBuffer buffer(capacity());
  for (int i = 0; i < 2 * buffer.Capacity(); ++i) {
    buffer.Push({2, 4});
  }
  EXPECT_EQ(buffer.Capacity(), std::max(capacity(), 1));
  EXPECT_EQ(buffer.Size(), std::max(capacity(), 1));
}

TEST_P(ClippingPredictorLevelBufferParametrization, CheckSizeAfterReset) {
  ClippingPredictorLevelBuffer buffer(capacity());
  buffer.Push({1, 1});
  buffer.Push({1, 1});
  buffer.Reset();
  EXPECT_EQ(buffer.Capacity(), std::max(capacity(), 1));
  EXPECT_EQ(buffer.Size(), 0);
  buffer.Push({1, 1});
  EXPECT_EQ(buffer.Capacity(), std::max(capacity(), 1));
  EXPECT_EQ(buffer.Size(), 1);
}

INSTANTIATE_TEST_SUITE_P(ClippingPredictorLevelBufferTest,
                         ClippingPredictorLevelBufferParametrization,
                         ::testing::Values(-1, 0, 1, 123));

TEST(ClippingPredictorLevelBufferTest, CheckMetricsAfterFullBuffer) {
  ClippingPredictorLevelBuffer buffer(/*capacity=*/2);
  buffer.Push({1, 2});
  buffer.Push({3, 6});
  EXPECT_THAT(buffer.ComputePartialMetrics(/*delay=*/0, /*num_items=*/1),
              Optional(Eq(ClippingPredictorLevelBuffer::Level{3, 6})));
  EXPECT_THAT(buffer.ComputePartialMetrics(/*delay=*/1, /*num_items=*/1),
              Optional(Eq(ClippingPredictorLevelBuffer::Level{1, 2})));
  EXPECT_THAT(buffer.ComputePartialMetrics(/*delay=*/0, /*num_items=*/2),
              Optional(Eq(ClippingPredictorLevelBuffer::Level{2, 6})));
}

TEST(ClippingPredictorLevelBufferTest, CheckMetricsAfterPushBeyondCapacity) {
  ClippingPredictorLevelBuffer buffer(/*capacity=*/2);
  buffer.Push({1, 1});
  buffer.Push({3, 6});
  buffer.Push({5, 10});
  buffer.Push({7, 14});
  buffer.Push({6, 12});
  EXPECT_THAT(buffer.ComputePartialMetrics(/*delay=*/0, /*num_items=*/1),
              Optional(Eq(ClippingPredictorLevelBuffer::Level{6, 12})));
  EXPECT_THAT(buffer.ComputePartialMetrics(/*delay=*/1, /*num_items=*/1),
              Optional(Eq(ClippingPredictorLevelBuffer::Level{7, 14})));
  EXPECT_THAT(buffer.ComputePartialMetrics(/*delay=*/0, /*num_items=*/2),
              Optional(Eq(ClippingPredictorLevelBuffer::Level{6.5f, 14})));
}

TEST(ClippingPredictorLevelBufferTest, CheckMetricsAfterTooFewItems) {
  ClippingPredictorLevelBuffer buffer(/*capacity=*/4);
  buffer.Push({1, 2});
  buffer.Push({3, 6});
  EXPECT_EQ(buffer.ComputePartialMetrics(/*delay=*/0, /*num_items=*/3),
            std::nullopt);
  EXPECT_EQ(buffer.ComputePartialMetrics(/*delay=*/2, /*num_items=*/1),
            std::nullopt);
}

TEST(ClippingPredictorLevelBufferTest, CheckMetricsAfterReset) {
  ClippingPredictorLevelBuffer buffer(/*capacity=*/2);
  buffer.Push({1, 2});
  buffer.Reset();
  buffer.Push({5, 10});
  buffer.Push({7, 14});
  EXPECT_THAT(buffer.ComputePartialMetrics(/*delay=*/0, /*num_items=*/1),
              Optional(Eq(ClippingPredictorLevelBuffer::Level{7, 14})));
  EXPECT_THAT(buffer.ComputePartialMetrics(/*delay=*/0, /*num_items=*/2),
              Optional(Eq(ClippingPredictorLevelBuffer::Level{6, 14})));
  EXPECT_THAT(buffer.ComputePartialMetrics(/*delay=*/1, /*num_items=*/1),
              Optional(Eq(ClippingPredictorLevelBuffer::Level{5, 10})));
}

}  // namespace
}  // namespace webrtc
