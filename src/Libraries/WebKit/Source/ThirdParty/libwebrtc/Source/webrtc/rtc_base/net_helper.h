/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 27, 2022.
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
#ifndef RTC_BASE_NET_HELPER_H_
#define RTC_BASE_NET_HELPER_H_

#include <string>

#include "absl/strings/string_view.h"
#include "rtc_base/system/rtc_export.h"

// This header contains helper functions and constants used by different types
// of transports.
namespace cricket {

RTC_EXPORT extern const char UDP_PROTOCOL_NAME[];
RTC_EXPORT extern const char TCP_PROTOCOL_NAME[];
extern const char SSLTCP_PROTOCOL_NAME[];
extern const char TLS_PROTOCOL_NAME[];

constexpr int kTcpHeaderSize = 20;
constexpr int kUdpHeaderSize = 8;

// Get the transport layer overhead per packet based on the protocol.
int GetProtocolOverhead(absl::string_view protocol);

}  // namespace cricket

#endif  // RTC_BASE_NET_HELPER_H_
