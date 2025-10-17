/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 6, 2023.
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
#include "rtc_base/data_rate_limiter.h"

namespace rtc {

bool DataRateLimiter::CanUse(size_t desired, double time) {
  return ((time > period_end_ && desired <= max_per_period_) ||
          (used_in_period_ + desired) <= max_per_period_);
}

void DataRateLimiter::Use(size_t used, double time) {
  if (time > period_end_) {
    period_start_ = time;
    period_end_ = time + period_length_;
    used_in_period_ = 0;
  }
  used_in_period_ += used;
}

}  // namespace rtc
