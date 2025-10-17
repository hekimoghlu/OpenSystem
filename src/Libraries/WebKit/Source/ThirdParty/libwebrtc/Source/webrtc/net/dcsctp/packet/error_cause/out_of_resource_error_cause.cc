/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 30, 2022.
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
#include "net/dcsctp/packet/error_cause/out_of_resource_error_cause.h"

#include <stdint.h>

#include <optional>
#include <vector>

#include "api/array_view.h"

namespace dcsctp {

// https://tools.ietf.org/html/rfc4960#section-3.3.10.4

//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |     Cause Code=4              |      Cause Length=4           |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
constexpr int OutOfResourceErrorCause::kType;

std::optional<OutOfResourceErrorCause> OutOfResourceErrorCause::Parse(
    rtc::ArrayView<const uint8_t> data) {
  if (!ParseTLV(data).has_value()) {
    return std::nullopt;
  }
  return OutOfResourceErrorCause();
}

void OutOfResourceErrorCause::SerializeTo(std::vector<uint8_t>& out) const {
  AllocateTLV(out);
}

std::string OutOfResourceErrorCause::ToString() const {
  return "Out Of Resource";
}

}  // namespace dcsctp
