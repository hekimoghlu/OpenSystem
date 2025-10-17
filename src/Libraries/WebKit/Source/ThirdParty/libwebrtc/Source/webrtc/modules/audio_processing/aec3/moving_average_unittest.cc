/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 9, 2021.
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
#include "modules/audio_processing/aec3/moving_average.h"

#include "test/gtest.h"

namespace webrtc {

TEST(MovingAverage, Average) {
  constexpr size_t num_elem = 4;
  constexpr size_t mem_len = 3;
  constexpr float e = 1e-6f;
  aec3::MovingAverage ma(num_elem, mem_len);
  std::array<float, num_elem> data1 = {1, 2, 3, 4};
  std::array<float, num_elem> data2 = {5, 1, 9, 7};
  std::array<float, num_elem> data3 = {3, 3, 5, 6};
  std::array<float, num_elem> data4 = {8, 4, 2, 1};
  std::array<float, num_elem> output;

  ma.Average(data1, output);
  EXPECT_NEAR(output[0], data1[0] / 3.0f, e);
  EXPECT_NEAR(output[1], data1[1] / 3.0f, e);
  EXPECT_NEAR(output[2], data1[2] / 3.0f, e);
  EXPECT_NEAR(output[3], data1[3] / 3.0f, e);

  ma.Average(data2, output);
  EXPECT_NEAR(output[0], (data1[0] + data2[0]) / 3.0f, e);
  EXPECT_NEAR(output[1], (data1[1] + data2[1]) / 3.0f, e);
  EXPECT_NEAR(output[2], (data1[2] + data2[2]) / 3.0f, e);
  EXPECT_NEAR(output[3], (data1[3] + data2[3]) / 3.0f, e);

  ma.Average(data3, output);
  EXPECT_NEAR(output[0], (data1[0] + data2[0] + data3[0]) / 3.0f, e);
  EXPECT_NEAR(output[1], (data1[1] + data2[1] + data3[1]) / 3.0f, e);
  EXPECT_NEAR(output[2], (data1[2] + data2[2] + data3[2]) / 3.0f, e);
  EXPECT_NEAR(output[3], (data1[3] + data2[3] + data3[3]) / 3.0f, e);

  ma.Average(data4, output);
  EXPECT_NEAR(output[0], (data2[0] + data3[0] + data4[0]) / 3.0f, e);
  EXPECT_NEAR(output[1], (data2[1] + data3[1] + data4[1]) / 3.0f, e);
  EXPECT_NEAR(output[2], (data2[2] + data3[2] + data4[2]) / 3.0f, e);
  EXPECT_NEAR(output[3], (data2[3] + data3[3] + data4[3]) / 3.0f, e);
}

TEST(MovingAverage, PassThrough) {
  constexpr size_t num_elem = 4;
  constexpr size_t mem_len = 1;
  constexpr float e = 1e-6f;
  aec3::MovingAverage ma(num_elem, mem_len);
  std::array<float, num_elem> data1 = {1, 2, 3, 4};
  std::array<float, num_elem> data2 = {5, 1, 9, 7};
  std::array<float, num_elem> data3 = {3, 3, 5, 6};
  std::array<float, num_elem> data4 = {8, 4, 2, 1};
  std::array<float, num_elem> output;

  ma.Average(data1, output);
  EXPECT_NEAR(output[0], data1[0], e);
  EXPECT_NEAR(output[1], data1[1], e);
  EXPECT_NEAR(output[2], data1[2], e);
  EXPECT_NEAR(output[3], data1[3], e);

  ma.Average(data2, output);
  EXPECT_NEAR(output[0], data2[0], e);
  EXPECT_NEAR(output[1], data2[1], e);
  EXPECT_NEAR(output[2], data2[2], e);
  EXPECT_NEAR(output[3], data2[3], e);

  ma.Average(data3, output);
  EXPECT_NEAR(output[0], data3[0], e);
  EXPECT_NEAR(output[1], data3[1], e);
  EXPECT_NEAR(output[2], data3[2], e);
  EXPECT_NEAR(output[3], data3[3], e);

  ma.Average(data4, output);
  EXPECT_NEAR(output[0], data4[0], e);
  EXPECT_NEAR(output[1], data4[1], e);
  EXPECT_NEAR(output[2], data4[2], e);
  EXPECT_NEAR(output[3], data4[3], e);
}

}  // namespace webrtc
