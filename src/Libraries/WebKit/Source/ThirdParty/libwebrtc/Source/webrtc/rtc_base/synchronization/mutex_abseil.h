/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 25, 2022.
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
#ifndef RTC_BASE_SYNCHRONIZATION_MUTEX_ABSEIL_H_
#define RTC_BASE_SYNCHRONIZATION_MUTEX_ABSEIL_H_

#include "absl/base/attributes.h"
#include "absl/synchronization/mutex.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {

class RTC_LOCKABLE MutexImpl final {
 public:
  MutexImpl() = default;
  MutexImpl(const MutexImpl&) = delete;
  MutexImpl& operator=(const MutexImpl&) = delete;

  void Lock() RTC_EXCLUSIVE_LOCK_FUNCTION() { mutex_.Lock(); }
  ABSL_MUST_USE_RESULT bool TryLock() RTC_EXCLUSIVE_TRYLOCK_FUNCTION(true) {
    return mutex_.TryLock();
  }
  void AssertHeld() const RTC_ASSERT_EXCLUSIVE_LOCK() {
#if RTC_DCHECK_IS_ON
    mutex_.AssertHeld();
#endif
  }
  void Unlock() RTC_UNLOCK_FUNCTION() { mutex_.Unlock(); }

 private:
  absl::Mutex mutex_;
};

}  // namespace webrtc

#endif  // RTC_BASE_SYNCHRONIZATION_MUTEX_ABSEIL_H_
