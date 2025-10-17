/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 15, 2024.
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
#ifndef SYSTEM_WRAPPERS_INCLUDE_CLOCK_H_
#define SYSTEM_WRAPPERS_INCLUDE_CLOCK_H_

#include <stdint.h>

#include <atomic>
#include <memory>

#include "api/units/timestamp.h"
#include "rtc_base/system/rtc_export.h"
#include "system_wrappers/include/ntp_time.h"

namespace webrtc {

// January 1970, in NTP seconds.
const uint32_t kNtpJan1970 = 2208988800UL;

// Magic NTP fractional unit.
const double kMagicNtpFractionalUnit = 4.294967296E+9;

// A clock interface that allows reading of absolute and relative timestamps.
class RTC_EXPORT Clock {
 public:
  virtual ~Clock() {}

  // Return a timestamp relative to an unspecified epoch.
  virtual Timestamp CurrentTime() = 0;
  int64_t TimeInMilliseconds() { return CurrentTime().ms(); }
  int64_t TimeInMicroseconds() { return CurrentTime().us(); }

  // Retrieve an NTP absolute timestamp (with an epoch of Jan 1, 1900).
  NtpTime CurrentNtpTime() { return ConvertTimestampToNtpTime(CurrentTime()); }
  int64_t CurrentNtpInMilliseconds() { return CurrentNtpTime().ToMs(); }

  // Converts between a relative timestamp returned by this clock, to NTP time.
  virtual NtpTime ConvertTimestampToNtpTime(Timestamp timestamp) = 0;
  int64_t ConvertTimestampToNtpTimeInMilliseconds(int64_t timestamp_ms) {
    return ConvertTimestampToNtpTime(Timestamp::Millis(timestamp_ms)).ToMs();
  }

  // Converts NtpTime to a Timestamp with UTC epoch.
  // A `Minus Infinity` Timestamp is returned if the NtpTime is invalid.
  static Timestamp NtpToUtc(NtpTime ntp_time) {
    if (!ntp_time.Valid()) {
      return Timestamp::MinusInfinity();
    }
    // Seconds since UTC epoch.
    int64_t time = ntp_time.seconds() - kNtpJan1970;
    // Microseconds since UTC epoch (not including NTP fraction)
    time = time * 1'000'000;
    // Fractions part of the NTP time, in microseconds.
    int64_t time_fraction =
        DivideRoundToNearest(int64_t{ntp_time.fractions()} * 1'000'000,
                             NtpTime::kFractionsPerSecond);
    return Timestamp::Micros(time + time_fraction);
  }

  // Returns an instance of the real-time system clock implementation.
  static Clock* GetRealTimeClock();
};

class SimulatedClock : public Clock {
 public:
  // The constructors assume an epoch of Jan 1, 1970.
  explicit SimulatedClock(int64_t initial_time_us);
  explicit SimulatedClock(Timestamp initial_time);
  ~SimulatedClock() override;

  // Return a timestamp with an epoch of Jan 1, 1970.
  Timestamp CurrentTime() override;

  NtpTime ConvertTimestampToNtpTime(Timestamp timestamp) override;

  // Advance the simulated clock with a given number of milliseconds or
  // microseconds.
  void AdvanceTimeMilliseconds(int64_t milliseconds);
  void AdvanceTimeMicroseconds(int64_t microseconds);
  void AdvanceTime(TimeDelta delta);

 private:
  // The time is read and incremented with relaxed order. Each thread will see
  // monotonically increasing time, and when threads post tasks or messages to
  // one another, the synchronization done as part of the message passing should
  // ensure that any causual chain of events on multiple threads also
  // corresponds to monotonically increasing time.
  std::atomic<int64_t> time_us_;
};

}  // namespace webrtc

#endif  // SYSTEM_WRAPPERS_INCLUDE_CLOCK_H_
