/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 17, 2023.
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
#ifndef RTC_BASE_DATA_RATE_LIMITER_H_
#define RTC_BASE_DATA_RATE_LIMITER_H_

#include <stddef.h>

#include "rtc_base/system/rtc_export.h"

namespace rtc {

// Limits the rate of use to a certain maximum quantity per period of
// time.  Use, for example, for simple bandwidth throttling.
//
// It's implemented like a diet plan: You have so many calories per
// day.  If you hit the limit, you can't eat any more until the next
// day.
class RTC_EXPORT DataRateLimiter {
 public:
  // For example, 100kb per second.
  DataRateLimiter(size_t max, double period)
      : max_per_period_(max),
        period_length_(period),
        used_in_period_(0),
        period_start_(0.0),
        period_end_(period) {}
  virtual ~DataRateLimiter() {}

  // Returns true if if the desired quantity is available in the
  // current period (< (max - used)).  Once the given time passes the
  // end of the period, used is set to zero and more use is available.
  bool CanUse(size_t desired, double time);
  // Increment the quantity used this period.  If past the end of a
  // period, a new period is started.
  void Use(size_t used, double time);

  size_t used_in_period() const { return used_in_period_; }

  size_t max_per_period() const { return max_per_period_; }

 private:
  size_t max_per_period_;
  double period_length_;
  size_t used_in_period_;
  double period_start_;
  double period_end_;
};
}  // namespace rtc

#endif  // RTC_BASE_DATA_RATE_LIMITER_H_
