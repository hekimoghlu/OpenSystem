/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 10, 2023.
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
#ifndef RTC_BASE_NUMERICS_PERCENTILE_FILTER_H_
#define RTC_BASE_NUMERICS_PERCENTILE_FILTER_H_

#include <stdint.h>

#include <iterator>
#include <set>

#include "rtc_base/checks.h"

namespace webrtc {

// Class to efficiently get the percentile value from a group of observations.
// The percentile is the value below which a given percentage of the
// observations fall.
template <typename T>
class PercentileFilter {
 public:
  // Construct filter. `percentile` should be between 0 and 1.
  explicit PercentileFilter(float percentile);

  // Insert one observation. The complexity of this operation is logarithmic in
  // the size of the container.
  void Insert(const T& value);

  // Remove one observation or return false if `value` doesn't exist in the
  // container. The complexity of this operation is logarithmic in the size of
  // the container.
  bool Erase(const T& value);

  // Get the percentile value. The complexity of this operation is constant.
  T GetPercentileValue() const;

  // Removes all the stored observations.
  void Reset();

 private:
  // Update iterator and index to point at target percentile value.
  void UpdatePercentileIterator();

  const float percentile_;
  std::multiset<T> set_;
  // Maintain iterator and index of current target percentile value.
  typename std::multiset<T>::iterator percentile_it_;
  int64_t percentile_index_;
};

template <typename T>
PercentileFilter<T>::PercentileFilter(float percentile)
    : percentile_(percentile),
      percentile_it_(set_.begin()),
      percentile_index_(0) {
  RTC_CHECK_GE(percentile, 0.0f);
  RTC_CHECK_LE(percentile, 1.0f);
}

template <typename T>
void PercentileFilter<T>::Insert(const T& value) {
  // Insert element at the upper bound.
  set_.insert(value);
  if (set_.size() == 1u) {
    // First element inserted - initialize percentile iterator and index.
    percentile_it_ = set_.begin();
    percentile_index_ = 0;
  } else if (value < *percentile_it_) {
    // If new element is before us, increment `percentile_index_`.
    ++percentile_index_;
  }
  UpdatePercentileIterator();
}

template <typename T>
bool PercentileFilter<T>::Erase(const T& value) {
  typename std::multiset<T>::const_iterator it = set_.lower_bound(value);
  // Ignore erase operation if the element is not present in the current set.
  if (it == set_.end() || *it != value)
    return false;
  if (it == percentile_it_) {
    // If same iterator, update to the following element. Index is not
    // affected.
    percentile_it_ = set_.erase(it);
  } else {
    set_.erase(it);
    // If erased element was before us, decrement `percentile_index_`.
    if (value <= *percentile_it_)
      --percentile_index_;
  }
  UpdatePercentileIterator();
  return true;
}

template <typename T>
void PercentileFilter<T>::UpdatePercentileIterator() {
  if (set_.empty())
    return;
  const int64_t index = static_cast<int64_t>(percentile_ * (set_.size() - 1));
  std::advance(percentile_it_, index - percentile_index_);
  percentile_index_ = index;
}

template <typename T>
T PercentileFilter<T>::GetPercentileValue() const {
  return set_.empty() ? 0 : *percentile_it_;
}

template <typename T>
void PercentileFilter<T>::Reset() {
  set_.clear();
  percentile_it_ = set_.begin();
  percentile_index_ = 0;
}
}  // namespace webrtc

#endif  // RTC_BASE_NUMERICS_PERCENTILE_FILTER_H_
