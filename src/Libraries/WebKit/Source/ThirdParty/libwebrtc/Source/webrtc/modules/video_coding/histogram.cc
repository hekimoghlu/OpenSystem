/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 16, 2025.
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
#include "modules/video_coding/histogram.h"

#include <algorithm>

#include "rtc_base/checks.h"

namespace webrtc {
namespace video_coding {
Histogram::Histogram(size_t num_buckets, size_t max_num_values) {
  RTC_DCHECK_GT(num_buckets, 0);
  RTC_DCHECK_GT(max_num_values, 0);
  buckets_.resize(num_buckets);
  values_.reserve(max_num_values);
  index_ = 0;
}

void Histogram::Add(size_t value) {
  value = std::min<size_t>(value, buckets_.size() - 1);
  if (index_ < values_.size()) {
    --buckets_[values_[index_]];
    RTC_DCHECK_LT(values_[index_], buckets_.size());
    values_[index_] = value;
  } else {
    values_.emplace_back(value);
  }

  ++buckets_[value];
  index_ = (index_ + 1) % values_.capacity();
}

size_t Histogram::InverseCdf(float probability) const {
  RTC_DCHECK_GE(probability, 0.f);
  RTC_DCHECK_LE(probability, 1.f);
  RTC_DCHECK_GT(values_.size(), 0ul);

  size_t bucket = 0;
  float accumulated_probability = 0;
  while (accumulated_probability < probability && bucket < buckets_.size()) {
    accumulated_probability +=
        static_cast<float>(buckets_[bucket]) / values_.size();
    ++bucket;
  }
  return bucket;
}

size_t Histogram::NumValues() const {
  return values_.size();
}

}  // namespace video_coding
}  // namespace webrtc
