/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 22, 2021.
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
#ifndef MODULES_PACING_INTERVAL_BUDGET_H_
#define MODULES_PACING_INTERVAL_BUDGET_H_

#include <stddef.h>
#include <stdint.h>

namespace webrtc {

// TODO(tschumim): Reflector IntervalBudget so that we can set a under- and
// over-use budget in ms.
class IntervalBudget {
 public:
  explicit IntervalBudget(int initial_target_rate_kbps);
  IntervalBudget(int initial_target_rate_kbps, bool can_build_up_underuse);
  void set_target_rate_kbps(int target_rate_kbps);

  // TODO(tschumim): Unify IncreaseBudget and UseBudget to one function.
  void IncreaseBudget(int64_t delta_time_ms);
  void UseBudget(size_t bytes);

  size_t bytes_remaining() const;
  double budget_ratio() const;
  int target_rate_kbps() const;

 private:
  int target_rate_kbps_;
  int64_t max_bytes_in_budget_;
  int64_t bytes_remaining_;
  bool can_build_up_underuse_;
};

}  // namespace webrtc

#endif  // MODULES_PACING_INTERVAL_BUDGET_H_
