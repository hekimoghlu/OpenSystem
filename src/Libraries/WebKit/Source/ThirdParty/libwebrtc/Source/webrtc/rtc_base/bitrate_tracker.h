/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 7, 2024.
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
#ifndef RTC_BASE_BITRATE_TRACKER_H_
#define RTC_BASE_BITRATE_TRACKER_H_

#include <stddef.h>
#include <stdint.h>

#include <optional>

#include "api/units/data_rate.h"
#include "api/units/data_size.h"
#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "rtc_base/rate_statistics.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {
// Class to estimate bitrates over running window.
// Timestamps used in Update(), Rate() and SetWindowSize() must never
// decrease for two consecutive calls.
// This class is thread unsafe.
class RTC_EXPORT BitrateTracker {
 public:
  // max_window_sizes = Maximum window size for the rate estimation.
  //                    Initial window size is set to this, but may be changed
  //                    to something lower by calling SetWindowSize().
  explicit BitrateTracker(TimeDelta max_window_size);

  BitrateTracker(const BitrateTracker&) = default;
  BitrateTracker(BitrateTracker&&) = default;
  BitrateTracker& operator=(const BitrateTracker&) = delete;
  BitrateTracker& operator=(BitrateTracker&&) = delete;

  ~BitrateTracker() = default;

  // Resets instance to original state.
  void Reset() { impl_.Reset(); }

  // Updates bitrate with a new data point, moving averaging window as needed.
  void Update(int64_t bytes, Timestamp now);
  void Update(DataSize size, Timestamp now) { Update(size.bytes(), now); }

  // Returns bitrate, moving averaging window as needed.
  // Returns nullopt when bitrate can't be measured.
  std::optional<DataRate> Rate(Timestamp now) const;

  // Update the size of the averaging window. The maximum allowed value for
  // `window_size` is `max_window_size` as supplied in the constructor.
  bool SetWindowSize(TimeDelta window_size, Timestamp now);

 private:
  RateStatistics impl_;
};
}  // namespace webrtc

#endif  // RTC_BASE_BITRATE_TRACKER_H_
