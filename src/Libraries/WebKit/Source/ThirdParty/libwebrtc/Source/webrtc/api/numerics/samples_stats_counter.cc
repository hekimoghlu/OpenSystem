/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 11, 2024.
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
#include "api/numerics/samples_stats_counter.h"

#include <algorithm>
#include <cmath>

#include "absl/algorithm/container.h"
#include "rtc_base/time_utils.h"

namespace webrtc {

SamplesStatsCounter::SamplesStatsCounter() = default;
SamplesStatsCounter::SamplesStatsCounter(size_t expected_samples_count) {
  samples_.reserve(expected_samples_count);
}

SamplesStatsCounter::~SamplesStatsCounter() = default;
SamplesStatsCounter::SamplesStatsCounter(const SamplesStatsCounter&) = default;
SamplesStatsCounter& SamplesStatsCounter::operator=(
    const SamplesStatsCounter&) = default;
SamplesStatsCounter::SamplesStatsCounter(SamplesStatsCounter&&) = default;
SamplesStatsCounter& SamplesStatsCounter::operator=(SamplesStatsCounter&&) =
    default;

void SamplesStatsCounter::AddSample(double value) {
  AddSample(StatsSample{value, Timestamp::Micros(rtc::TimeMicros())});
}

void SamplesStatsCounter::AddSample(StatsSample sample) {
  stats_.AddSample(sample.value);
  samples_.push_back(sample);
  sorted_ = false;
}

void SamplesStatsCounter::AddSamples(const SamplesStatsCounter& other) {
  stats_.MergeStatistics(other.stats_);
  samples_.insert(samples_.end(), other.samples_.begin(), other.samples_.end());
  sorted_ = false;
}

double SamplesStatsCounter::GetPercentile(double percentile) {
  RTC_DCHECK(!IsEmpty());
  RTC_CHECK_GE(percentile, 0);
  RTC_CHECK_LE(percentile, 1);
  if (!sorted_) {
    absl::c_sort(samples_, [](const StatsSample& a, const StatsSample& b) {
      return a.value < b.value;
    });
    sorted_ = true;
  }
  const double raw_rank = percentile * (samples_.size() - 1);
  double int_part;
  double fract_part = std::modf(raw_rank, &int_part);
  size_t rank = static_cast<size_t>(int_part);
  if (fract_part >= 1.0) {
    // It can happen due to floating point calculation error.
    rank++;
    fract_part -= 1.0;
  }

  RTC_DCHECK_GE(rank, 0);
  RTC_DCHECK_LT(rank, samples_.size());
  RTC_DCHECK_GE(fract_part, 0);
  RTC_DCHECK_LT(fract_part, 1);
  RTC_DCHECK(rank + fract_part == raw_rank);

  const double low = samples_[rank].value;
  const double high = samples_[std::min(rank + 1, samples_.size() - 1)].value;
  return low + fract_part * (high - low);
}

SamplesStatsCounter operator*(const SamplesStatsCounter& counter,
                              double value) {
  SamplesStatsCounter out;
  for (const auto& sample : counter.GetTimedSamples()) {
    out.AddSample(
        SamplesStatsCounter::StatsSample{sample.value * value, sample.time});
  }
  return out;
}

SamplesStatsCounter operator/(const SamplesStatsCounter& counter,
                              double value) {
  SamplesStatsCounter out;
  for (const auto& sample : counter.GetTimedSamples()) {
    out.AddSample(
        SamplesStatsCounter::StatsSample{sample.value / value, sample.time});
  }
  return out;
}

}  // namespace webrtc
