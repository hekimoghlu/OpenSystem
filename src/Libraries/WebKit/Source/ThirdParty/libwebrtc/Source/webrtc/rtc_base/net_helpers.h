/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 8, 2023.
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
#ifndef RTC_BASE_NET_HELPERS_H_
#define RTC_BASE_NET_HELPERS_H_

#if defined(WEBRTC_POSIX)
#include <sys/socket.h>  // IWYU pragma: export
#elif WEBRTC_WIN
#include <winsock2.h>  // NOLINT

#include "rtc_base/win32.h"
#endif

#include "absl/strings/string_view.h"
#include "rtc_base/system/rtc_export.h"

namespace rtc {

// rtc namespaced wrappers for inet_ntop and inet_pton so we can avoid
// the windows-native versions of these.
const char* inet_ntop(int af, const void* src, char* dst, socklen_t size);
int inet_pton(int af, absl::string_view src, void* dst);

}  // namespace rtc

#endif  // RTC_BASE_NET_HELPERS_H_
