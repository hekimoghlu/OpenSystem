/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 19, 2022.
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
#ifndef RTC_BASE_SYNCHRONIZATION_MUTEX_H_
#define RTC_BASE_SYNCHRONIZATION_MUTEX_H_

#include <atomic>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "rtc_base/checks.h"
#include "rtc_base/thread_annotations.h"

#if defined(WEBRTC_ABSL_MUTEX)
#include "rtc_base/synchronization/mutex_abseil.h"  // nogncheck
#elif defined(WEBRTC_WIN)
#include "rtc_base/synchronization/mutex_critical_section.h"
#elif defined(WEBRTC_POSIX)
#include "rtc_base/synchronization/mutex_pthread.h"
#else
#error Unsupported platform.
#endif

namespace webrtc {

// The Mutex guarantees exclusive access and aims to follow Abseil semantics
// (i.e. non-reentrant etc).
class RTC_LOCKABLE Mutex final {
 public:
  Mutex() = default;
  Mutex(const Mutex&) = delete;
  Mutex& operator=(const Mutex&) = delete;

  void Lock() RTC_EXCLUSIVE_LOCK_FUNCTION() { impl_.Lock(); }
  ABSL_MUST_USE_RESULT bool TryLock() RTC_EXCLUSIVE_TRYLOCK_FUNCTION(true) {
    return impl_.TryLock();
  }
  // Return immediately if this thread holds the mutex, or RTC_DCHECK_IS_ON==0.
  // Otherwise, may report an error (typically by crashing with a diagnostic),
  // or may return immediately.
  void AssertHeld() const RTC_ASSERT_EXCLUSIVE_LOCK() { impl_.AssertHeld(); }
  void Unlock() RTC_UNLOCK_FUNCTION() { impl_.Unlock(); }

 private:
  MutexImpl impl_;
};

// MutexLock, for serializing execution through a scope.
class RTC_SCOPED_LOCKABLE MutexLock final {
 public:
  MutexLock(const MutexLock&) = delete;
  MutexLock& operator=(const MutexLock&) = delete;

  explicit MutexLock(Mutex* mutex) RTC_EXCLUSIVE_LOCK_FUNCTION(mutex)
      : mutex_(mutex) {
    mutex->Lock();
  }
  ~MutexLock() RTC_UNLOCK_FUNCTION() { mutex_->Unlock(); }

 private:
  Mutex* mutex_;
};

}  // namespace webrtc

#endif  // RTC_BASE_SYNCHRONIZATION_MUTEX_H_
