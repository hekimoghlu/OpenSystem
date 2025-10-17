/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 26, 2023.
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
#include "rtc_base/numerics/sample_counter.h"

#include <cstdint>
#include <limits>
#include <optional>

#include "rtc_base/checks.h"
#include "rtc_base/numerics/safe_conversions.h"

namespace rtc {

SampleCounter::SampleCounter() = default;
SampleCounter::~SampleCounter() = default;

void SampleCounter::Add(int sample) {
  if (sum_ > 0) {
    RTC_DCHECK_LE(sample, std::numeric_limits<int64_t>::max() - sum_);
  } else {
    RTC_DCHECK_GE(sample, std::numeric_limits<int64_t>::min() - sum_);
  }
  sum_ += sample;
  ++num_samples_;
  if (!max_ || sample > *max_) {
    max_ = sample;
  }
  if (!min_ || sample < *min_) {
    min_ = sample;
  }
}

void SampleCounter::Add(const SampleCounter& other) {
  if (sum_ > 0) {
    RTC_DCHECK_LE(other.sum_, std::numeric_limits<int64_t>::max() - sum_);
  } else {
    RTC_DCHECK_GE(other.sum_, std::numeric_limits<int64_t>::min() - sum_);
  }
  sum_ += other.sum_;
  RTC_DCHECK_LE(other.num_samples_,
                std::numeric_limits<int64_t>::max() - num_samples_);
  num_samples_ += other.num_samples_;
  if (other.max_ && (!max_ || *max_ < *other.max_))
    max_ = other.max_;
  if (other.min_ && (!min_ || *min_ > *other.min_))
    min_ = other.min_;
}

std::optional<int> SampleCounter::Avg(int64_t min_required_samples) const {
  RTC_DCHECK_GT(min_required_samples, 0);
  if (num_samples_ < min_required_samples)
    return std::nullopt;
  return rtc::dchecked_cast<int>(sum_ / num_samples_);
}

std::optional<int> SampleCounter::Max() const {
  return max_;
}

std::optional<int> SampleCounter::Min() const {
  return min_;
}

std::optional<int64_t> SampleCounter::Sum(int64_t min_required_samples) const {
  RTC_DCHECK_GT(min_required_samples, 0);
  if (num_samples_ < min_required_samples)
    return std::nullopt;
  return sum_;
}

int64_t SampleCounter::NumSamples() const {
  return num_samples_;
}

void SampleCounter::Reset() {
  *this = {};
}

SampleCounterWithVariance::SampleCounterWithVariance() = default;
SampleCounterWithVariance::~SampleCounterWithVariance() = default;

std::optional<int64_t> SampleCounterWithVariance::Variance(
    int64_t min_required_samples) const {
  RTC_DCHECK_GT(min_required_samples, 0);
  if (num_samples_ < min_required_samples)
    return std::nullopt;
  // E[(x-mean)^2] = E[x^2] - mean^2
  int64_t mean = sum_ / num_samples_;
  return sum_squared_ / num_samples_ - mean * mean;
}

void SampleCounterWithVariance::Add(int sample) {
  SampleCounter::Add(sample);
  // Prevent overflow in squaring.
  RTC_DCHECK_GT(sample, std::numeric_limits<int32_t>::min());
  RTC_DCHECK_LE(int64_t{sample} * sample,
                std::numeric_limits<int64_t>::max() - sum_squared_);
  sum_squared_ += int64_t{sample} * sample;
}

void SampleCounterWithVariance::Add(const SampleCounterWithVariance& other) {
  SampleCounter::Add(other);
  RTC_DCHECK_LE(other.sum_squared_,
                std::numeric_limits<int64_t>::max() - sum_squared_);
  sum_squared_ += other.sum_squared_;
}

void SampleCounterWithVariance::Reset() {
  *this = {};
}

}  // namespace rtc
