/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 29, 2023.
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
#ifndef RTC_BASE_FREQUENCY_TRACKER_H_
#define RTC_BASE_FREQUENCY_TRACKER_H_

#include <stddef.h>
#include <stdint.h>

#include <optional>

#include "api/units/frequency.h"
#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "rtc_base/rate_statistics.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {
// Class to estimate frequency (e.g. frame rate) over running window.
// Timestamps used in Update() and Rate() must never decrease for two
// consecutive calls.
// This class is thread unsafe.
class RTC_EXPORT FrequencyTracker {
 public:
  explicit FrequencyTracker(TimeDelta window_size);

  FrequencyTracker(const FrequencyTracker&) = default;
  FrequencyTracker(FrequencyTracker&&) = default;
  FrequencyTracker& operator=(const FrequencyTracker&) = delete;
  FrequencyTracker& operator=(FrequencyTracker&&) = delete;

  ~FrequencyTracker() = default;

  // Reset instance to original state.
  void Reset() { impl_.Reset(); }

  // Update rate with a new data point, moving averaging window as needed.
  void Update(int64_t count, Timestamp now);
  void Update(Timestamp now) { Update(1, now); }

  // Returns rate, moving averaging window as needed.
  // Returns nullopt when rate can't be measured.
  std::optional<Frequency> Rate(Timestamp now) const;

 private:
  RateStatistics impl_;
};
}  // namespace webrtc

#endif  // RTC_BASE_FREQUENCY_TRACKER_H_
