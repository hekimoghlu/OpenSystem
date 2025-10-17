/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 18, 2025.
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
#include "rtc_base/net_helper.h"

#include "absl/strings/string_view.h"

namespace cricket {

const char UDP_PROTOCOL_NAME[] = "udp";
const char TCP_PROTOCOL_NAME[] = "tcp";
const char SSLTCP_PROTOCOL_NAME[] = "ssltcp";
const char TLS_PROTOCOL_NAME[] = "tls";

int GetProtocolOverhead(absl::string_view protocol) {
  if (protocol == TCP_PROTOCOL_NAME || protocol == SSLTCP_PROTOCOL_NAME) {
    return kTcpHeaderSize;
  } else if (protocol == UDP_PROTOCOL_NAME) {
    return kUdpHeaderSize;
  } else {
    // TODO(srte): We should crash on unexpected input and handle TLS correctly.
    return 8;
  }
}

}  // namespace cricket
