/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 1, 2022.
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
#ifndef RTC_BASE_NUMERICS_MOVING_AVERAGE_H_
#define RTC_BASE_NUMERICS_MOVING_AVERAGE_H_

#include <stddef.h>
#include <stdint.h>

#include <optional>
#include <vector>

namespace rtc {

// Calculates average over fixed size window. If there are less than window
// size elements, calculates average of all inserted so far elements.
//
class MovingAverage {
 public:
  // Maximum supported window size is 2^32 - 1.
  explicit MovingAverage(size_t window_size);
  ~MovingAverage();
  // MovingAverage is neither copyable nor movable.
  MovingAverage(const MovingAverage&) = delete;
  MovingAverage& operator=(const MovingAverage&) = delete;

  // Adds new sample. If the window is full, the oldest element is pushed out.
  void AddSample(int sample);

  // Returns rounded down average of last `window_size` elements or all
  // elements if there are not enough of them. Returns nullopt if there were
  // no elements added.
  std::optional<int> GetAverageRoundedDown() const;

  // Same as above but rounded to the closest integer.
  std::optional<int> GetAverageRoundedToClosest() const;

  // Returns unrounded average over the window.
  std::optional<double> GetUnroundedAverage() const;

  // Resets to the initial state before any elements were added.
  void Reset();

  // Returns number of elements in the window.
  size_t Size() const;

 private:
  // Total number of samples added to the class since last reset.
  size_t count_ = 0;
  // Sum of the samples in the moving window.
  int64_t sum_ = 0;
  // Circular buffer for all the samples in the moving window.
  // Size is always `window_size`
  std::vector<int> history_;
};

}  // namespace rtc
#endif  // RTC_BASE_NUMERICS_MOVING_AVERAGE_H_
