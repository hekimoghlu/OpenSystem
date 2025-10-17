/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 21, 2024.
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
#include "rtc_base/net_helpers.h"

#include <memory>
#include <string>

#include "absl/strings/string_view.h"

#if defined(WEBRTC_WIN)
#include <ws2spi.h>
#include <ws2tcpip.h>

#endif
#if defined(WEBRTC_POSIX) && !defined(__native_client__)
#include <arpa/inet.h>
#endif  // defined(WEBRTC_POSIX) && !defined(__native_client__)

namespace rtc {

const char* inet_ntop(int af, const void* src, char* dst, socklen_t size) {
#if defined(WEBRTC_WIN)
  return win32_inet_ntop(af, src, dst, size);
#else
  return ::inet_ntop(af, src, dst, size);
#endif
}

int inet_pton(int af, absl::string_view src, void* dst) {
  std::string src_str(src);
#if defined(WEBRTC_WIN)
  return win32_inet_pton(af, src_str.c_str(), dst);
#else
  return ::inet_pton(af, src_str.c_str(), dst);
#endif
}
}  // namespace rtc
