/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 30, 2024.
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
#include "modules/audio_coding/neteq/underrun_optimizer.h"

#include <algorithm>

namespace webrtc {

namespace {

constexpr int kDelayBuckets = 100;
constexpr int kBucketSizeMs = 20;

}  // namespace

UnderrunOptimizer::UnderrunOptimizer(const TickTimer* tick_timer,
                                     int histogram_quantile,
                                     int forget_factor,
                                     std::optional<int> start_forget_weight,
                                     std::optional<int> resample_interval_ms)
    : tick_timer_(tick_timer),
      histogram_(kDelayBuckets, forget_factor, start_forget_weight),
      histogram_quantile_(histogram_quantile),
      resample_interval_ms_(resample_interval_ms) {}

void UnderrunOptimizer::Update(int relative_delay_ms) {
  std::optional<int> histogram_update;
  if (resample_interval_ms_) {
    if (!resample_stopwatch_) {
      resample_stopwatch_ = tick_timer_->GetNewStopwatch();
    }
    if (static_cast<int>(resample_stopwatch_->ElapsedMs()) >
        *resample_interval_ms_) {
      histogram_update = max_delay_in_interval_ms_;
      resample_stopwatch_ = tick_timer_->GetNewStopwatch();
      max_delay_in_interval_ms_ = 0;
    }
    max_delay_in_interval_ms_ =
        std::max(max_delay_in_interval_ms_, relative_delay_ms);
  } else {
    histogram_update = relative_delay_ms;
  }
  if (!histogram_update) {
    return;
  }

  const int index = *histogram_update / kBucketSizeMs;
  if (index < histogram_.NumBuckets()) {
    // Maximum delay to register is 2000 ms.
    histogram_.Add(index);
  }
  int bucket_index = histogram_.Quantile(histogram_quantile_);
  optimal_delay_ms_ = (1 + bucket_index) * kBucketSizeMs;
}

void UnderrunOptimizer::Reset() {
  histogram_.Reset();
  resample_stopwatch_.reset();
  max_delay_in_interval_ms_ = 0;
  optimal_delay_ms_.reset();
}

}  // namespace webrtc
