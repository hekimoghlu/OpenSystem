/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 24, 2022.
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
#include "test/drifting_clock.h"

#include "rtc_base/checks.h"

namespace webrtc {
namespace test {
constexpr float DriftingClock::kNoDrift;

DriftingClock::DriftingClock(Clock* clock, float speed)
    : clock_(clock), drift_(speed - 1.0f), start_time_(clock_->CurrentTime()) {
  RTC_CHECK(clock);
  RTC_CHECK_GT(speed, 0.0f);
}

TimeDelta DriftingClock::Drift() const {
  auto now = clock_->CurrentTime();
  RTC_DCHECK_GE(now, start_time_);
  return (now - start_time_) * drift_;
}

Timestamp DriftingClock::Drift(Timestamp timestamp) const {
  return timestamp + Drift() / 1000.;
}

NtpTime DriftingClock::Drift(NtpTime ntp_time) const {
  // NTP precision is 1/2^32 seconds, i.e. 2^32 ntp fractions = 1 second.
  const double kNtpFracPerMicroSecond = 4294.967296;  // = 2^32 / 10^6

  uint64_t total_fractions = static_cast<uint64_t>(ntp_time);
  total_fractions += Drift().us() * kNtpFracPerMicroSecond;
  return NtpTime(total_fractions);
}

}  // namespace test
}  // namespace webrtc
