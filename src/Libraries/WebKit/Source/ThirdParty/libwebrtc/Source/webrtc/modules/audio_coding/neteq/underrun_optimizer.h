/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 27, 2025.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_UNDERRUN_OPTIMIZER_H_
#define MODULES_AUDIO_CODING_NETEQ_UNDERRUN_OPTIMIZER_H_

#include <memory>
#include <optional>

#include "api/neteq/tick_timer.h"
#include "modules/audio_coding/neteq/histogram.h"

namespace webrtc {

// Estimates probability of buffer underrun due to late packet arrival.
// The optimal delay is decided such that the probability of underrun is lower
// than 1 - `histogram_quantile`.
class UnderrunOptimizer {
 public:
  UnderrunOptimizer(const TickTimer* tick_timer,
                    int histogram_quantile,
                    int forget_factor,
                    std::optional<int> start_forget_weight,
                    std::optional<int> resample_interval_ms);

  void Update(int relative_delay_ms);

  std::optional<int> GetOptimalDelayMs() const { return optimal_delay_ms_; }

  void Reset();

 private:
  const TickTimer* tick_timer_;
  Histogram histogram_;
  const int histogram_quantile_;  // In Q30.
  const std::optional<int> resample_interval_ms_;
  std::unique_ptr<TickTimer::Stopwatch> resample_stopwatch_;
  int max_delay_in_interval_ms_ = 0;
  std::optional<int> optimal_delay_ms_;
};

}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_UNDERRUN_OPTIMIZER_H_
