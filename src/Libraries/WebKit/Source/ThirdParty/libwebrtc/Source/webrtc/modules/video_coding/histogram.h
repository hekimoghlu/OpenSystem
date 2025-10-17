/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 24, 2024.
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
#ifndef MODULES_VIDEO_CODING_HISTOGRAM_H_
#define MODULES_VIDEO_CODING_HISTOGRAM_H_

#include <cstddef>
#include <vector>

namespace webrtc {
namespace video_coding {
class Histogram {
 public:
  // A discrete histogram where every bucket with range [0, num_buckets).
  // Values greater or equal to num_buckets will be placed in the last bucket.
  Histogram(size_t num_buckets, size_t max_num_values);

  // Add a value to the histogram. If there already is max_num_values in the
  // histogram then the oldest value will be replaced with the new value.
  void Add(size_t value);

  // Calculates how many buckets have to be summed in order to accumulate at
  // least the given probability.
  size_t InverseCdf(float probability) const;

  // How many values that make up this histogram.
  size_t NumValues() const;

 private:
  // A circular buffer that holds the values that make up the histogram.
  std::vector<size_t> values_;
  std::vector<size_t> buckets_;
  size_t index_;
};

}  // namespace video_coding
}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_HISTOGRAM_H_
