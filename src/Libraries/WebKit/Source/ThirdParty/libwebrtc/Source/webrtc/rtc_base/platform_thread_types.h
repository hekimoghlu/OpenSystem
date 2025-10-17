/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 11, 2024.
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
#ifndef RTC_BASE_PLATFORM_THREAD_TYPES_H_
#define RTC_BASE_PLATFORM_THREAD_TYPES_H_

// clang-format off
// clang formating would change include order.
#if defined(WEBRTC_WIN)
// Include winsock2.h before including <windows.h> to maintain consistency with
// win32.h. To include win32.h directly, it must be broken out into its own
// build target.
#include <winsock2.h>
#include <windows.h>
#elif defined(WEBRTC_FUCHSIA)
#include <zircon/types.h>
#include <zircon/process.h>
#elif defined(WEBRTC_POSIX)
#include <pthread.h>
#include <unistd.h>
#if defined(WEBRTC_MAC)
#include <pthread_spis.h>
#endif
#endif
// clang-format on

namespace rtc {
#if defined(WEBRTC_WIN)
typedef DWORD PlatformThreadId;
typedef DWORD PlatformThreadRef;
#elif defined(WEBRTC_FUCHSIA)
typedef zx_handle_t PlatformThreadId;
typedef zx_handle_t PlatformThreadRef;
#elif defined(WEBRTC_POSIX)
typedef pid_t PlatformThreadId;
typedef pthread_t PlatformThreadRef;
#endif

// Retrieve the ID of the current thread.
PlatformThreadId CurrentThreadId();

// Retrieves a reference to the current thread. On Windows, this is the same
// as CurrentThreadId. On other platforms it's the pthread_t returned by
// pthread_self().
PlatformThreadRef CurrentThreadRef();

// Compares two thread identifiers for equality.
bool IsThreadRefEqual(const PlatformThreadRef& a, const PlatformThreadRef& b);

// Sets the current thread name.
void SetCurrentThreadName(const char* name);

}  // namespace rtc

#endif  // RTC_BASE_PLATFORM_THREAD_TYPES_H_
