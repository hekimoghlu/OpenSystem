/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 28, 2025.
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
#ifndef RTC_BASE_NUMERICS_SAMPLE_COUNTER_H_
#define RTC_BASE_NUMERICS_SAMPLE_COUNTER_H_

#include <stdint.h>

#include <optional>

namespace rtc {

// Simple utility class for counting basic statistics (max./avg./variance) on
// stream of samples.
class SampleCounter {
 public:
  SampleCounter();
  ~SampleCounter();
  void Add(int sample);
  std::optional<int> Avg(int64_t min_required_samples) const;
  std::optional<int> Max() const;
  std::optional<int> Min() const;
  std::optional<int64_t> Sum(int64_t min_required_samples) const;
  int64_t NumSamples() const;
  void Reset();
  // Adds all the samples from the `other` SampleCounter as if they were all
  // individually added using `Add(int)` method.
  void Add(const SampleCounter& other);

 protected:
  int64_t sum_ = 0;
  int64_t num_samples_ = 0;
  std::optional<int> max_;
  std::optional<int> min_;
};

class SampleCounterWithVariance : public SampleCounter {
 public:
  SampleCounterWithVariance();
  ~SampleCounterWithVariance();
  void Add(int sample);
  std::optional<int64_t> Variance(int64_t min_required_samples) const;
  void Reset();
  // Adds all the samples from the `other` SampleCounter as if they were all
  // individually added using `Add(int)` method.
  void Add(const SampleCounterWithVariance& other);

 private:
  int64_t sum_squared_ = 0;
};

}  // namespace rtc
#endif  // RTC_BASE_NUMERICS_SAMPLE_COUNTER_H_
