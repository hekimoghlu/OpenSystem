/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 7, 2023.
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
#include "modules/video_coding/timing/frame_delay_variation_kalman_filter.h"

#include "test/gtest.h"

namespace webrtc {
namespace {

// This test verifies that the initial filter state (link bandwidth, link
// propagation delay) is such that a frame of size zero would take no time to
// propagate.
TEST(FrameDelayVariationKalmanFilterTest,
     InitializedFilterWithZeroSizeFrameTakesNoTimeToPropagate) {
  FrameDelayVariationKalmanFilter filter;

  // A zero-sized frame...
  double frame_size_variation_bytes = 0.0;

  // ...should take no time to propagate due to it's size...
  EXPECT_EQ(filter.GetFrameDelayVariationEstimateSizeBased(
                frame_size_variation_bytes),
            0.0);

  // ...and no time due to the initial link propagation delay being zero.
  EXPECT_EQ(
      filter.GetFrameDelayVariationEstimateTotal(frame_size_variation_bytes),
      0.0);
}

// TODO(brandtr): Look into if there is a factor 1000 missing here? It seems
// unreasonable to have an initial link bandwidth of 512 _mega_bits per second?
TEST(FrameDelayVariationKalmanFilterTest,
     InitializedFilterWithSmallSizeFrameTakesFixedTimeToPropagate) {
  FrameDelayVariationKalmanFilter filter;

  // A 1000-byte frame...
  double frame_size_variation_bytes = 1000.0;
  // ...should take around `1000.0 / (512e3 / 8.0) = 0.015625 ms` to transmit.
  double expected_frame_delay_variation_estimate_ms = 1000.0 / (512e3 / 8.0);

  EXPECT_EQ(filter.GetFrameDelayVariationEstimateSizeBased(
                frame_size_variation_bytes),
            expected_frame_delay_variation_estimate_ms);
  EXPECT_EQ(
      filter.GetFrameDelayVariationEstimateTotal(frame_size_variation_bytes),
      expected_frame_delay_variation_estimate_ms);
}

TEST(FrameDelayVariationKalmanFilterTest,
     NegativeNoiseVarianceDoesNotUpdateFilter) {
  FrameDelayVariationKalmanFilter filter;

  // Negative variance...
  double var_noise = -0.1;
  filter.PredictAndUpdate(/*frame_delay_variation_ms=*/3,
                          /*frame_size_variation_bytes=*/200.0,
                          /*max_frame_size_bytes=*/2000, var_noise);

  // ...does _not_ update the filter.
  EXPECT_EQ(filter.GetFrameDelayVariationEstimateTotal(
                /*frame_size_variation_bytes=*/0.0),
            0.0);

  // Positive variance...
  var_noise = 0.1;
  filter.PredictAndUpdate(/*frame_delay_variation_ms=*/3,
                          /*frame_size_variation_bytes=*/200.0,
                          /*max_frame_size_bytes=*/2000, var_noise);

  // ...does update the filter.
  EXPECT_GT(filter.GetFrameDelayVariationEstimateTotal(
                /*frame_size_variation_bytes=*/0.0),
            0.0);
}

TEST(FrameDelayVariationKalmanFilterTest,
     VerifyConvergenceWithAlternatingDeviations) {
  FrameDelayVariationKalmanFilter filter;

  // One frame every 33 ms.
  int framerate_fps = 30;
  // Let's assume approximately 10% delay variation.
  double frame_delay_variation_ms = 3;
  // With a bitrate of 512 kbps, each frame will be around 2000 bytes.
  double max_frame_size_bytes = 2000;
  // And again, let's assume 10% size deviation.
  double frame_size_variation_bytes = 200;
  double var_noise = 0.1;
  int test_duration_s = 60;

  for (int i = 0; i < test_duration_s * framerate_fps; ++i) {
    // For simplicity, assume alternating variations.
    double sign = (i % 2 == 0) ? 1.0 : -1.0;
    filter.PredictAndUpdate(sign * frame_delay_variation_ms,
                            sign * frame_size_variation_bytes,
                            max_frame_size_bytes, var_noise);
  }

  // Verify that the filter has converged within a margin of 0.1 ms.
  EXPECT_NEAR(
      filter.GetFrameDelayVariationEstimateTotal(frame_size_variation_bytes),
      frame_delay_variation_ms, 0.1);
}

}  // namespace
}  // namespace webrtc
