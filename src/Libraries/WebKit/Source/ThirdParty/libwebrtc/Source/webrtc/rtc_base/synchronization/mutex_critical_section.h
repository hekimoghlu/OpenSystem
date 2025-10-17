/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 6, 2025.
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
#ifndef RTC_BASE_SYNCHRONIZATION_MUTEX_CRITICAL_SECTION_H_
#define RTC_BASE_SYNCHRONIZATION_MUTEX_CRITICAL_SECTION_H_

#if defined(WEBRTC_WIN)
// clang-format off
// clang formating would change include order.

// Include winsock2.h before including <windows.h> to maintain consistency with
// win32.h. To include win32.h directly, it must be broken out into its own
// build target.
#include <winsock2.h>
#include <windows.h>
#include <sal.h>  // must come after windows headers.
// clang-format on

#include "absl/base/attributes.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {

class RTC_LOCKABLE MutexImpl final {
 public:
  MutexImpl() { InitializeCriticalSection(&critical_section_); }
  MutexImpl(const MutexImpl&) = delete;
  MutexImpl& operator=(const MutexImpl&) = delete;
  ~MutexImpl() { DeleteCriticalSection(&critical_section_); }

  void Lock() RTC_EXCLUSIVE_LOCK_FUNCTION() {
    EnterCriticalSection(&critical_section_);
  }
  ABSL_MUST_USE_RESULT bool TryLock() RTC_EXCLUSIVE_TRYLOCK_FUNCTION(true) {
    return TryEnterCriticalSection(&critical_section_) != FALSE;
  }
  void AssertHeld() const RTC_ASSERT_EXCLUSIVE_LOCK() {}
  void Unlock() RTC_UNLOCK_FUNCTION() {
    LeaveCriticalSection(&critical_section_);
  }

 private:
  CRITICAL_SECTION critical_section_;
};

}  // namespace webrtc

#endif  // #if defined(WEBRTC_WIN)
#endif  // RTC_BASE_SYNCHRONIZATION_MUTEX_CRITICAL_SECTION_H_
