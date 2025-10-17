/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 15, 2023.
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
#include "rtc_base/fake_clock.h"

#include "rtc_base/checks.h"
#include "rtc_base/thread.h"

namespace rtc {

int64_t FakeClock::TimeNanos() const {
  webrtc::MutexLock lock(&lock_);
  return time_ns_;
}

void FakeClock::SetTime(webrtc::Timestamp new_time) {
  webrtc::MutexLock lock(&lock_);
  RTC_DCHECK(new_time.us() * 1000 >= time_ns_);
  time_ns_ = new_time.us() * 1000;
}

void FakeClock::AdvanceTime(webrtc::TimeDelta delta) {
  webrtc::MutexLock lock(&lock_);
  time_ns_ += delta.ns();
}

void ThreadProcessingFakeClock::SetTime(webrtc::Timestamp time) {
  clock_.SetTime(time);
  // If message queues are waiting in a socket select() with a timeout provided
  // by the OS, they should wake up and dispatch all messages that are ready.
  ThreadManager::ProcessAllMessageQueuesForTesting();
}

void ThreadProcessingFakeClock::AdvanceTime(webrtc::TimeDelta delta) {
  clock_.AdvanceTime(delta);
  ThreadManager::ProcessAllMessageQueuesForTesting();
}

ScopedBaseFakeClock::ScopedBaseFakeClock() {
  prev_clock_ = SetClockForTesting(this);
}

ScopedBaseFakeClock::~ScopedBaseFakeClock() {
  SetClockForTesting(prev_clock_);
}

ScopedFakeClock::ScopedFakeClock() {
  prev_clock_ = SetClockForTesting(this);
}

ScopedFakeClock::~ScopedFakeClock() {
  SetClockForTesting(prev_clock_);
}

}  // namespace rtc
