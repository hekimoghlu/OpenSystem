/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 4, 2023.
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
#include "rtc_base/numerics/moving_average.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "rtc_base/checks.h"

namespace rtc {

MovingAverage::MovingAverage(size_t window_size) : history_(window_size, 0) {
  // Limit window size to avoid overflow.
  RTC_DCHECK_LE(window_size, (int64_t{1} << 32) - 1);
}
MovingAverage::~MovingAverage() = default;

void MovingAverage::AddSample(int sample) {
  count_++;
  size_t index = count_ % history_.size();
  if (count_ > history_.size())
    sum_ -= history_[index];
  sum_ += sample;
  history_[index] = sample;
}

std::optional<int> MovingAverage::GetAverageRoundedDown() const {
  if (count_ == 0)
    return std::nullopt;
  return sum_ / Size();
}

std::optional<int> MovingAverage::GetAverageRoundedToClosest() const {
  if (count_ == 0)
    return std::nullopt;
  return (sum_ + Size() / 2) / Size();
}

std::optional<double> MovingAverage::GetUnroundedAverage() const {
  if (count_ == 0)
    return std::nullopt;
  return sum_ / static_cast<double>(Size());
}

void MovingAverage::Reset() {
  count_ = 0;
  sum_ = 0;
}

size_t MovingAverage::Size() const {
  return std::min(count_, history_.size());
}
}  // namespace rtc
