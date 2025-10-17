/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 23, 2022.
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
#include "rtc_tools/frame_analyzer/linear_least_squares.h"

#include <cstdint>

#include "test/gtest.h"

namespace webrtc {
namespace test {

TEST(LinearLeastSquares, ScalarIdentityOneObservation) {
  IncrementalLinearLeastSquares lls;
  lls.AddObservations({{1}}, {{1}});
  EXPECT_EQ(std::vector<std::vector<double>>({{1.0}}), lls.GetBestSolution());
}

TEST(LinearLeastSquares, ScalarIdentityTwoObservationsOneCall) {
  IncrementalLinearLeastSquares lls;
  lls.AddObservations({{1, 2}}, {{1, 2}});
  EXPECT_EQ(std::vector<std::vector<double>>({{1.0}}), lls.GetBestSolution());
}

TEST(LinearLeastSquares, ScalarIdentityTwoObservationsTwoCalls) {
  IncrementalLinearLeastSquares lls;
  lls.AddObservations({{1}}, {{1}});
  lls.AddObservations({{2}}, {{2}});
  EXPECT_EQ(std::vector<std::vector<double>>({{1.0}}), lls.GetBestSolution());
}

TEST(LinearLeastSquares, MatrixIdentityOneObservation) {
  IncrementalLinearLeastSquares lls;
  lls.AddObservations({{1, 2}, {3, 4}}, {{1, 2}, {3, 4}});
  EXPECT_EQ(std::vector<std::vector<double>>({{1.0, 0.0}, {0.0, 1.0}}),
            lls.GetBestSolution());
}

TEST(LinearLeastSquares, MatrixManyObservations) {
  IncrementalLinearLeastSquares lls;
  // Test that we can find the solution of the overspecified equation system:
  // [1, 2] [1, 3] = [5,  11]
  // [3, 4] [2, 4]   [11, 25]
  // [5, 6]          [17, 39]
  lls.AddObservations({{1}, {2}}, {{5}, {11}});
  lls.AddObservations({{3}, {4}}, {{11}, {25}});
  lls.AddObservations({{5}, {6}}, {{17}, {39}});

  const std::vector<std::vector<double>> result = lls.GetBestSolution();
  // We allow some numerical flexibility here.
  EXPECT_DOUBLE_EQ(1.0, result[0][0]);
  EXPECT_DOUBLE_EQ(2.0, result[0][1]);
  EXPECT_DOUBLE_EQ(3.0, result[1][0]);
  EXPECT_DOUBLE_EQ(4.0, result[1][1]);
}

TEST(LinearLeastSquares, MatrixVectorOneObservation) {
  IncrementalLinearLeastSquares lls;
  // Test that we can find the solution of the overspecified equation system:
  // [1, 2] [1] = [5]
  // [3, 4] [2]   [11]
  // [5, 6]       [17]
  lls.AddObservations({{1, 3, 5}, {2, 4, 6}}, {{5, 11, 17}});

  const std::vector<std::vector<double>> result = lls.GetBestSolution();
  // We allow some numerical flexibility here.
  EXPECT_DOUBLE_EQ(1.0, result[0][0]);
  EXPECT_DOUBLE_EQ(2.0, result[0][1]);
}

TEST(LinearLeastSquares, LinearLeastSquaresNonPerfectSolution) {
  IncrementalLinearLeastSquares lls;
  // Test that we can find the non-perfect solution of the overspecified
  // equation system:
  // [1] [20] = [21]
  // [2]        [39]
  // [3]        [60]
  // [2]        [41]
  // [1]        [19]
  lls.AddObservations({{1, 2, 3, 2, 1}}, {{21, 39, 60, 41, 19}});

  EXPECT_DOUBLE_EQ(20.0, lls.GetBestSolution()[0][0]);
}

}  // namespace test
}  // namespace webrtc
