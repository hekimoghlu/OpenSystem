/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 29, 2024.
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
#ifndef RTC_BASE_ONE_TIME_EVENT_H_
#define RTC_BASE_ONE_TIME_EVENT_H_

#include "rtc_base/synchronization/mutex.h"

namespace webrtc {
// Provides a simple way to perform an operation (such as logging) one
// time in a certain scope.
// Example:
//   OneTimeEvent firstFrame;
//   ...
//   if (firstFrame()) {
//     RTC_LOG(LS_INFO) << "This is the first frame".
//   }
class OneTimeEvent {
 public:
  OneTimeEvent() {}
  bool operator()() {
    MutexLock lock(&mutex_);
    if (happened_) {
      return false;
    }
    happened_ = true;
    return true;
  }

 private:
  bool happened_ = false;
  Mutex mutex_;
};

// A non-thread-safe, ligher-weight version of the OneTimeEvent class.
class ThreadUnsafeOneTimeEvent {
 public:
  ThreadUnsafeOneTimeEvent() {}
  bool operator()() {
    if (happened_) {
      return false;
    }
    happened_ = true;
    return true;
  }

 private:
  bool happened_ = false;
};

}  // namespace webrtc

#endif  // RTC_BASE_ONE_TIME_EVENT_H_
