/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 6, 2024.
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
#ifndef TEST_DRIFTING_CLOCK_H_
#define TEST_DRIFTING_CLOCK_H_

#include <stdint.h>

#include "system_wrappers/include/clock.h"
#include "system_wrappers/include/ntp_time.h"

namespace webrtc {
namespace test {
class DriftingClock : public Clock {
 public:
  static constexpr float kNoDrift = 1.0f;

  DriftingClock(Clock* clock, float speed);

  static constexpr float PercentsFaster(float percent) {
    return 1.0f + percent / 100.0f;
  }
  static constexpr float PercentsSlower(float percent) {
    return 1.0f - percent / 100.0f;
  }

  Timestamp CurrentTime() override { return Drift(clock_->CurrentTime()); }
  NtpTime ConvertTimestampToNtpTime(Timestamp timestamp) override {
    return Drift(clock_->ConvertTimestampToNtpTime(timestamp));
  }

 private:
  TimeDelta Drift() const;
  Timestamp Drift(Timestamp timestamp) const;
  NtpTime Drift(NtpTime ntp_time) const;

  Clock* const clock_;
  const float drift_;
  const Timestamp start_time_;
};
}  // namespace test
}  // namespace webrtc

#endif  // TEST_DRIFTING_CLOCK_H_
