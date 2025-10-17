/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 11, 2023.
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
#include "modules/video_coding/timing/decode_time_percentile_filter.h"

#include <cstdint>

namespace webrtc {

namespace {

// The first kIgnoredSampleCount samples will be ignored.
const int kIgnoredSampleCount = 5;
// Return the `kPercentile` value in RequiredDecodeTimeMs().
const float kPercentile = 0.95f;
// The window size in ms.
const int64_t kTimeLimitMs = 10000;

}  // anonymous namespace

DecodeTimePercentileFilter::DecodeTimePercentileFilter()
    : ignored_sample_count_(0), filter_(kPercentile) {}
DecodeTimePercentileFilter::~DecodeTimePercentileFilter() = default;

void DecodeTimePercentileFilter::AddTiming(int64_t decode_time_ms,
                                           int64_t now_ms) {
  // Ignore the first `kIgnoredSampleCount` samples.
  if (ignored_sample_count_ < kIgnoredSampleCount) {
    ++ignored_sample_count_;
    return;
  }

  // Insert new decode time value.
  filter_.Insert(decode_time_ms);
  history_.emplace(decode_time_ms, now_ms);

  // Pop old decode time values.
  while (!history_.empty() &&
         now_ms - history_.front().sample_time_ms > kTimeLimitMs) {
    filter_.Erase(history_.front().decode_time_ms);
    history_.pop();
  }
}

// Get the 95th percentile observed decode time within a time window.
int64_t DecodeTimePercentileFilter::RequiredDecodeTimeMs() const {
  return filter_.GetPercentileValue();
}

DecodeTimePercentileFilter::Sample::Sample(int64_t decode_time_ms,
                                           int64_t sample_time_ms)
    : decode_time_ms(decode_time_ms), sample_time_ms(sample_time_ms) {}

}  // namespace webrtc
