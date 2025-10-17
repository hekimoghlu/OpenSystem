/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 2, 2025.
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
#include "net/dcsctp/packet/error_cause/cookie_received_while_shutting_down_cause.h"

#include <stdint.h>

#include <optional>
#include <vector>

#include "api/array_view.h"

namespace dcsctp {

// https://tools.ietf.org/html/rfc4960#section-3.3.10.10

//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |     Cause Code=10              |      Cause Length=4          |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
constexpr int CookieReceivedWhileShuttingDownCause::kType;

std::optional<CookieReceivedWhileShuttingDownCause>
CookieReceivedWhileShuttingDownCause::Parse(
    rtc::ArrayView<const uint8_t> data) {
  if (!ParseTLV(data).has_value()) {
    return std::nullopt;
  }
  return CookieReceivedWhileShuttingDownCause();
}

void CookieReceivedWhileShuttingDownCause::SerializeTo(
    std::vector<uint8_t>& out) const {
  AllocateTLV(out);
}

std::string CookieReceivedWhileShuttingDownCause::ToString() const {
  return "Cookie Received While Shutting Down";
}
}  // namespace dcsctp
