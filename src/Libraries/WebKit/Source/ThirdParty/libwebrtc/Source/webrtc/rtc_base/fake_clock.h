/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 5, 2022.
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
#ifndef RTC_BASE_FAKE_CLOCK_H_
#define RTC_BASE_FAKE_CLOCK_H_

#include <stdint.h>

#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "rtc_base/synchronization/mutex.h"
#include "rtc_base/thread_annotations.h"
#include "rtc_base/time_utils.h"

namespace rtc {

// Fake clock for use with unit tests, which does not tick on its own.
// Starts at time 0.
//
// TODO(deadbeef): Unify with webrtc::SimulatedClock.
class FakeClock : public ClockInterface {
 public:
  FakeClock() = default;
  FakeClock(const FakeClock&) = delete;
  FakeClock& operator=(const FakeClock&) = delete;
  ~FakeClock() override = default;

  // ClockInterface implementation.
  int64_t TimeNanos() const override;

  // Methods that can be used by the test to control the time.

  // Should only be used to set a time in the future.
  void SetTime(webrtc::Timestamp new_time);

  void AdvanceTime(webrtc::TimeDelta delta);

 private:
  mutable webrtc::Mutex lock_;
  int64_t time_ns_ RTC_GUARDED_BY(lock_) = 0;
};

class ThreadProcessingFakeClock : public ClockInterface {
 public:
  int64_t TimeNanos() const override { return clock_.TimeNanos(); }
  void SetTime(webrtc::Timestamp time);
  void AdvanceTime(webrtc::TimeDelta delta);

 private:
  FakeClock clock_;
};

// Helper class that sets itself as the global clock in its constructor and
// unsets it in its destructor.
class ScopedBaseFakeClock : public FakeClock {
 public:
  ScopedBaseFakeClock();
  ~ScopedBaseFakeClock() override;

 private:
  ClockInterface* prev_clock_;
};

// TODO(srte): Rename this to reflect that it also does thread processing.
class ScopedFakeClock : public ThreadProcessingFakeClock {
 public:
  ScopedFakeClock();
  ~ScopedFakeClock() override;

 private:
  ClockInterface* prev_clock_;
};

}  // namespace rtc

#endif  // RTC_BASE_FAKE_CLOCK_H_
