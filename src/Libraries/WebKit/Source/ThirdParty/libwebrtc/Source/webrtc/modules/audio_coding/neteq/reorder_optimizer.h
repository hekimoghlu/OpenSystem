/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 16, 2022.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_REORDER_OPTIMIZER_H_
#define MODULES_AUDIO_CODING_NETEQ_REORDER_OPTIMIZER_H_

#include <optional>

#include "modules/audio_coding/neteq/histogram.h"

namespace webrtc {

// Calculates an optimal delay to reduce the chance of missing reordered
// packets. The delay/loss trade-off can be tune using the `ms_per_loss_percent`
// parameter.
class ReorderOptimizer {
 public:
  ReorderOptimizer(int forget_factor,
                   int ms_per_loss_percent,
                   std::optional<int> start_forget_weight);

  void Update(int relative_delay_ms, bool reordered, int base_delay_ms);

  std::optional<int> GetOptimalDelayMs() const { return optimal_delay_ms_; }

  void Reset();

 private:
  int MinimizeCostFunction(int base_delay_ms) const;

  Histogram histogram_;
  const int ms_per_loss_percent_;
  std::optional<int> optimal_delay_ms_;
};

}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_REORDER_OPTIMIZER_H_
