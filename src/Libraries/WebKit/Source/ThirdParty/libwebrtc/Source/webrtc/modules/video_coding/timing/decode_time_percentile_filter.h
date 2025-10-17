/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 30, 2024.
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
#ifndef MODULES_VIDEO_CODING_TIMING_DECODE_TIME_PERCENTILE_FILTER_H_
#define MODULES_VIDEO_CODING_TIMING_DECODE_TIME_PERCENTILE_FILTER_H_

#include <queue>

#include "rtc_base/numerics/percentile_filter.h"

namespace webrtc {

// The `DecodeTimePercentileFilter` filters the actual per-frame decode times
// and provides an estimate for the 95th percentile of those decode times. This
// estimate can be used to determine how large the "decode delay term" should be
// when determining the render timestamp for a frame.
class DecodeTimePercentileFilter {
 public:
  DecodeTimePercentileFilter();
  ~DecodeTimePercentileFilter();

  // Add a new decode time to the filter.
  void AddTiming(int64_t new_decode_time_ms, int64_t now_ms);

  // Get the required decode time in ms. It is the 95th percentile observed
  // decode time within a time window.
  int64_t RequiredDecodeTimeMs() const;

 private:
  struct Sample {
    Sample(int64_t decode_time_ms, int64_t sample_time_ms);
    int64_t decode_time_ms;
    int64_t sample_time_ms;
  };

  // The number of samples ignored so far.
  int ignored_sample_count_;
  // Queue with history of latest decode time values.
  std::queue<Sample> history_;
  // `filter_` contains the same values as `history_`, but in a data structure
  // that allows efficient retrieval of the percentile value.
  PercentileFilter<int64_t> filter_;
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_TIMING_DECODE_TIME_PERCENTILE_FILTER_H_
