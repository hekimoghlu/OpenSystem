/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 14, 2024.
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
#ifndef RTC_BASE_WIN32_H_
#define RTC_BASE_WIN32_H_

#ifndef WEBRTC_WIN
#error "Only #include this header in Windows builds"
#endif

// Make sure we don't get min/max macros
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <winsock2.h>

// Must be after winsock2.h.
#include <windows.h>

typedef int socklen_t;

#ifndef SECURITY_MANDATORY_LABEL_AUTHORITY
// Add defines that we use if we are compiling against older sdks
#define SECURITY_MANDATORY_MEDIUM_RID (0x00002000L)
#define TokenIntegrityLevel static_cast<TOKEN_INFORMATION_CLASS>(0x19)
typedef struct _TOKEN_MANDATORY_LABEL {
  SID_AND_ATTRIBUTES Label;
} TOKEN_MANDATORY_LABEL, *PTOKEN_MANDATORY_LABEL;
#endif  // SECURITY_MANDATORY_LABEL_AUTHORITY

#undef SetPort

namespace rtc {

const char* win32_inet_ntop(int af, const void* src, char* dst, socklen_t size);
int win32_inet_pton(int af, const char* src, void* dst);

}  // namespace rtc

#endif  // RTC_BASE_WIN32_H_
