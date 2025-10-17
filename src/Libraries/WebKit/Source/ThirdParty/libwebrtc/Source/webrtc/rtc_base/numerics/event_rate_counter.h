/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 16, 2024.
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
#ifndef RTC_BASE_NUMERICS_EVENT_RATE_COUNTER_H_
#define RTC_BASE_NUMERICS_EVENT_RATE_COUNTER_H_

#include <cstdint>

#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "rtc_base/numerics/sample_stats.h"

namespace webrtc {

// Calculates statistics based on events. For example for computing frame rates.
// Note that it doesn't provide any running statistics or reset funcitonality,
// so it's mostly useful for end of call statistics.
class EventRateCounter {
 public:
  // Adds an event based on it's `event_time` for correct updates of the
  // interval statistics, each event must be added past the previous events.
  void AddEvent(Timestamp event_time);
  // Adds the events from `other`. Note that the interval stats won't be
  // recalculated, only merged, so this is not equivalent to if the events would
  // have been added to the same counter from the start.
  void AddEvents(EventRateCounter other);
  bool IsEmpty() const;
  // Average number of events per second. Defaults to 0 for no events and NAN
  // for one event.
  double Rate() const;
  SampleStats<TimeDelta>& interval() { return interval_; }
  TimeDelta TotalDuration() const;
  int Count() const { return event_count_; }

 private:
  Timestamp first_time_ = Timestamp::PlusInfinity();
  Timestamp last_time_ = Timestamp::MinusInfinity();
  int64_t event_count_ = 0;
  SampleStats<TimeDelta> interval_;
};
}  // namespace webrtc
#endif  // RTC_BASE_NUMERICS_EVENT_RATE_COUNTER_H_
