/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 10, 2022.
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
#ifndef RTC_BASE_NUMERICS_HISTOGRAM_PERCENTILE_COUNTER_H_
#define RTC_BASE_NUMERICS_HISTOGRAM_PERCENTILE_COUNTER_H_

#include <stddef.h>
#include <stdint.h>

#include <map>
#include <optional>
#include <vector>

namespace rtc {
// Calculates percentiles on the stream of data. Use `Add` methods to add new
// values. Use `GetPercentile` to get percentile of the currently added values.
class HistogramPercentileCounter {
 public:
  // Values below `long_tail_boundary` are stored as the histogram in an array.
  // Values above - in a map.
  explicit HistogramPercentileCounter(uint32_t long_tail_boundary);
  ~HistogramPercentileCounter();
  void Add(uint32_t value);
  void Add(uint32_t value, size_t count);
  void Add(const HistogramPercentileCounter& other);
  // Argument should be from 0 to 1.
  std::optional<uint32_t> GetPercentile(float fraction);

 private:
  std::vector<size_t> histogram_low_;
  std::map<uint32_t, size_t> histogram_high_;
  const uint32_t long_tail_boundary_;
  size_t total_elements_;
  size_t total_elements_low_;
};
}  // namespace rtc
#endif  // RTC_BASE_NUMERICS_HISTOGRAM_PERCENTILE_COUNTER_H_
