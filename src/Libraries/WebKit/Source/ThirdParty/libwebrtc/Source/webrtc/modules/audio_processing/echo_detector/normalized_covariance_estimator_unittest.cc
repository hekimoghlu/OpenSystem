/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 20, 2023.
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
#include "modules/audio_processing/echo_detector/normalized_covariance_estimator.h"

#include "test/gtest.h"

namespace webrtc {

TEST(NormalizedCovarianceEstimatorTests, IdenticalSignalTest) {
  NormalizedCovarianceEstimator test_estimator;
  for (size_t i = 0; i < 10000; i++) {
    test_estimator.Update(1.f, 0.f, 1.f, 1.f, 0.f, 1.f);
    test_estimator.Update(-1.f, 0.f, 1.f, -1.f, 0.f, 1.f);
  }
  // A normalized covariance value close to 1 is expected.
  EXPECT_NEAR(1.f, test_estimator.normalized_cross_correlation(), 0.01f);
  test_estimator.Clear();
  EXPECT_EQ(0.f, test_estimator.normalized_cross_correlation());
}

TEST(NormalizedCovarianceEstimatorTests, OppositeSignalTest) {
  NormalizedCovarianceEstimator test_estimator;
  // Insert the same value many times.
  for (size_t i = 0; i < 10000; i++) {
    test_estimator.Update(1.f, 0.f, 1.f, -1.f, 0.f, 1.f);
    test_estimator.Update(-1.f, 0.f, 1.f, 1.f, 0.f, 1.f);
  }
  // A normalized covariance value close to -1 is expected.
  EXPECT_NEAR(-1.f, test_estimator.normalized_cross_correlation(), 0.01f);
}

}  // namespace webrtc
