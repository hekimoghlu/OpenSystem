/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 1, 2021.
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
#ifndef MODULES_RTP_RTCP_SOURCE_NTP_TIME_UTIL_H_
#define MODULES_RTP_RTCP_SOURCE_NTP_TIME_UTIL_H_

#include <stdint.h>

#include "api/units/time_delta.h"
#include "rtc_base/numerics/safe_conversions.h"
#include "system_wrappers/include/ntp_time.h"

namespace webrtc {

// Helper function for compact ntp representation:
// RFC 3550, Section 4. Time Format.
// Wallclock time is represented using the timestamp format of
// the Network Time Protocol (NTP).
// ...
// In some fields where a more compact representation is
// appropriate, only the middle 32 bits are used; that is, the low 16
// bits of the integer part and the high 16 bits of the fractional part.
inline uint32_t CompactNtp(NtpTime ntp) {
  return (ntp.seconds() << 16) | (ntp.fractions() >> 16);
}

// Converts interval to compact ntp (1/2^16 seconds) resolution.
// Negative values converted to 0, Overlarge values converted to max uint32_t.
uint32_t SaturatedToCompactNtp(TimeDelta delta);

// Convert interval to the NTP time resolution (1/2^32 seconds ~= 0.2 ns).
// For deltas with absolute value larger than 35 minutes result is unspecified.
inline constexpr int64_t ToNtpUnits(TimeDelta delta) {
  // For better precision `delta` is taken with best TimeDelta precision (us),
  // then multiplaction and conversion to seconds are swapped to avoid float
  // arithmetic.
  // 2^31 us ~= 35.8 minutes.
  return (rtc::saturated_cast<int32_t>(delta.us()) * (int64_t{1} << 32)) /
         1'000'000;
}

// Converts interval from compact ntp (1/2^16 seconds) resolution to TimeDelta.
// This interval can be up to ~9.1 hours (2^15 seconds).
// Values close to 2^16 seconds are considered negative.
TimeDelta CompactNtpIntervalToTimeDelta(uint32_t compact_ntp_interval);

// Converts interval from compact ntp (1/2^16 seconds) resolution to TimeDelta.
// This interval can be up to ~9.1 hours (2^15 seconds).
// Values close to 2^16 seconds are considered negative and are converted to
// minimum value of 1ms.
TimeDelta CompactNtpRttToTimeDelta(uint32_t compact_ntp_interval);

}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_SOURCE_NTP_TIME_UTIL_H_
