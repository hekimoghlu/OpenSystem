/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 11, 2023.
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
#include "modules/audio_processing/echo_detector/mean_variance_estimator.h"

#include "test/gtest.h"

namespace webrtc {

TEST(MeanVarianceEstimatorTests, InsertTwoValues) {
  MeanVarianceEstimator test_estimator;
  // Insert two values.
  test_estimator.Update(3.f);
  test_estimator.Update(5.f);

  EXPECT_GT(test_estimator.mean(), 0.f);
  EXPECT_GT(test_estimator.std_deviation(), 0.f);
  // Test Clear method
  test_estimator.Clear();
  EXPECT_EQ(test_estimator.mean(), 0.f);
  EXPECT_EQ(test_estimator.std_deviation(), 0.f);
}

TEST(MeanVarianceEstimatorTests, InsertZeroes) {
  MeanVarianceEstimator test_estimator;
  // Insert the same value many times.
  for (size_t i = 0; i < 20000; i++) {
    test_estimator.Update(0.f);
  }
  EXPECT_EQ(test_estimator.mean(), 0.f);
  EXPECT_EQ(test_estimator.std_deviation(), 0.f);
}

TEST(MeanVarianceEstimatorTests, ConstantValueTest) {
  MeanVarianceEstimator test_estimator;
  for (size_t i = 0; i < 20000; i++) {
    test_estimator.Update(3.f);
  }
  // The mean should be close to three, and the standard deviation should be
  // close to zero.
  EXPECT_NEAR(3.0f, test_estimator.mean(), 0.01f);
  EXPECT_NEAR(0.0f, test_estimator.std_deviation(), 0.01f);
}

TEST(MeanVarianceEstimatorTests, AlternatingValueTest) {
  MeanVarianceEstimator test_estimator;
  for (size_t i = 0; i < 20000; i++) {
    test_estimator.Update(1.f);
    test_estimator.Update(-1.f);
  }
  // The mean should be close to zero, and the standard deviation should be
  // close to one.
  EXPECT_NEAR(0.0f, test_estimator.mean(), 0.01f);
  EXPECT_NEAR(1.0f, test_estimator.std_deviation(), 0.01f);
}

}  // namespace webrtc
