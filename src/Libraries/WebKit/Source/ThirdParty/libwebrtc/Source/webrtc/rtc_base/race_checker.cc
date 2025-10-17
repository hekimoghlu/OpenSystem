/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 30, 2022.
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
#include "rtc_base/race_checker.h"

namespace rtc {

RaceChecker::RaceChecker() {}

// Note that the implementation here is in itself racy, but we pretend it does
// not matter because we want this useful in release builds without having to
// pay the cost of using atomics. A race hitting the race checker is likely to
// cause access_count_ to diverge from zero and therefore cause the ThreadRef
// comparison to fail, signaling a race, although it may not be in the exact
// spot where a race *first* appeared in the code we're trying to protect. There
// is also a chance that an actual race is missed, however the probability of
// that has been considered small enough to be an acceptable trade off.
bool RaceChecker::Acquire() const {
  const PlatformThreadRef current_thread = CurrentThreadRef();
  // Set new accessing thread if this is a new use.
  const int current_access_count = access_count_;
  access_count_ = access_count_ + 1;
  if (current_access_count == 0)
    accessing_thread_ = current_thread;
  // If this is being used concurrently this check will fail for the second
  // thread entering since it won't set the thread. Recursive use of checked
  // methods are OK since the accessing thread remains the same.
  const PlatformThreadRef accessing_thread = accessing_thread_;
  return IsThreadRefEqual(accessing_thread, current_thread);
}

void RaceChecker::Release() const {
  access_count_ = access_count_ - 1;
}

namespace internal {
RaceCheckerScope::RaceCheckerScope(const RaceChecker* race_checker)
    : race_checker_(race_checker), race_check_ok_(race_checker->Acquire()) {}

bool RaceCheckerScope::RaceDetected() const {
  return !race_check_ok_;
}

RaceCheckerScope::~RaceCheckerScope() {
  race_checker_->Release();
}

}  // namespace internal
}  // namespace rtc
