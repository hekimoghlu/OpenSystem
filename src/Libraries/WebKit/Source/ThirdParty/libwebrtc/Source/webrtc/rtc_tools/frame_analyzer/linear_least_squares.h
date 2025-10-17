/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 11, 2024.
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
#ifndef RTC_TOOLS_FRAME_ANALYZER_LINEAR_LEAST_SQUARES_H_
#define RTC_TOOLS_FRAME_ANALYZER_LINEAR_LEAST_SQUARES_H_

#include <stdint.h>

#include <optional>
#include <valarray>
#include <vector>

namespace webrtc {
namespace test {

// This class is used for finding a matrix b that roughly solves the equation:
// y = x * b. This is generally impossible to do exactly, so the problem is
// rephrased as finding the matrix b that minimizes the difference:
// |y - x * b|^2. Calling multiple AddObservations() is equivalent to
// concatenating the observation vectors and calling AddObservations() once. The
// reason for doing it incrementally is that we can't store the raw YUV values
// for a whole video file in memory at once. This class has a constant memory
// footprint, regardless how may times AddObservations() is called.
class IncrementalLinearLeastSquares {
 public:
  IncrementalLinearLeastSquares();
  ~IncrementalLinearLeastSquares();

  // Add a number of observations. The subvectors of x and y must have the same
  // length.
  void AddObservations(const std::vector<std::vector<uint8_t>>& x,
                       const std::vector<std::vector<uint8_t>>& y);

  // Calculate and return the best linear solution, given the observations so
  // far.
  std::vector<std::vector<double>> GetBestSolution() const;

 private:
  // Running sum of x^T * x.
  std::optional<std::valarray<std::valarray<uint64_t>>> sum_xx;
  // Running sum of x^T * y.
  std::optional<std::valarray<std::valarray<uint64_t>>> sum_xy;
};

}  // namespace test
}  // namespace webrtc

#endif  // RTC_TOOLS_FRAME_ANALYZER_LINEAR_LEAST_SQUARES_H_
