/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 14, 2022.
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
#include "system_wrappers/include/clock.h"

#include "api/units/time_delta.h"
#include "test/gtest.h"

namespace webrtc {

TEST(ClockTest, NtpTime) {
  Clock* clock = Clock::GetRealTimeClock();

  // To ensure the test runs correctly even on a heavily loaded system, do not
  // compare the seconds/fractions and millisecond values directly. Instead,
  // we check that the NTP time is between the "milliseconds" values returned
  // right before and right after the call.
  // The comparison includes 1 ms of margin to account for the rounding error in
  // the conversion.
  int64_t milliseconds_lower_bound = clock->CurrentNtpInMilliseconds();
  NtpTime ntp_time = clock->CurrentNtpTime();
  int64_t milliseconds_upper_bound = clock->CurrentNtpInMilliseconds();
  EXPECT_GT(milliseconds_lower_bound / 1000, kNtpJan1970);
  EXPECT_LE(milliseconds_lower_bound - 1, ntp_time.ToMs());
  EXPECT_GE(milliseconds_upper_bound + 1, ntp_time.ToMs());
}

TEST(ClockTest, NtpToUtc) {
  Clock* clock = Clock::GetRealTimeClock();
  NtpTime ntp = clock->CurrentNtpTime();
  Timestamp a = Clock::NtpToUtc(ntp);
  Timestamp b = Timestamp::Millis(ntp.ToMs() - int64_t{kNtpJan1970} * 1000);
  TimeDelta d = a - b;
  EXPECT_LT(d.Abs(), TimeDelta::Millis(1));
}

}  // namespace webrtc
