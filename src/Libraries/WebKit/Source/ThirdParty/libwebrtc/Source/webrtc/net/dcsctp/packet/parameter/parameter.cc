/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 13, 2023.
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
#include "net/dcsctp/packet/parameter/parameter.h"

#include <stddef.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "api/array_view.h"
#include "net/dcsctp/common/math.h"
#include "net/dcsctp/packet/bounded_byte_reader.h"
#include "net/dcsctp/packet/parameter/add_incoming_streams_request_parameter.h"
#include "net/dcsctp/packet/parameter/add_outgoing_streams_request_parameter.h"
#include "net/dcsctp/packet/parameter/forward_tsn_supported_parameter.h"
#include "net/dcsctp/packet/parameter/heartbeat_info_parameter.h"
#include "net/dcsctp/packet/parameter/incoming_ssn_reset_request_parameter.h"
#include "net/dcsctp/packet/parameter/outgoing_ssn_reset_request_parameter.h"
#include "net/dcsctp/packet/parameter/reconfiguration_response_parameter.h"
#include "net/dcsctp/packet/parameter/ssn_tsn_reset_request_parameter.h"
#include "net/dcsctp/packet/parameter/state_cookie_parameter.h"
#include "net/dcsctp/packet/parameter/supported_extensions_parameter.h"
#include "net/dcsctp/packet/tlv_trait.h"
#include "rtc_base/logging.h"
#include "rtc_base/strings/string_builder.h"

namespace dcsctp {

constexpr size_t kParameterHeaderSize = 4;

Parameters::Builder& Parameters::Builder::Add(const Parameter& p) {
  // https://tools.ietf.org/html/rfc4960#section-3.2.1
  // "If the length of the parameter is not a multiple of 4 bytes, the sender
  // pads the parameter at the end (i.e., after the Parameter Value field) with
  // all zero bytes."
  if (data_.size() % 4 != 0) {
    data_.resize(RoundUpTo4(data_.size()));
  }

  p.SerializeTo(data_);
  return *this;
}

std::vector<ParameterDescriptor> Parameters::descriptors() const {
  rtc::ArrayView<const uint8_t> span(data_);
  std::vector<ParameterDescriptor> result;
  while (!span.empty()) {
    BoundedByteReader<kParameterHeaderSize> header(span);
    uint16_t type = header.Load16<0>();
    uint16_t length = header.Load16<2>();
    result.emplace_back(type, span.subview(0, length));
    size_t length_with_padding = RoundUpTo4(length);
    if (length_with_padding > span.size()) {
      break;
    }
    span = span.subview(length_with_padding);
  }
  return result;
}

std::optional<Parameters> Parameters::Parse(
    rtc::ArrayView<const uint8_t> data) {
  // Validate the parameter descriptors
  rtc::ArrayView<const uint8_t> span(data);
  while (!span.empty()) {
    if (span.size() < kParameterHeaderSize) {
      RTC_DLOG(LS_WARNING) << "Insufficient parameter length";
      return std::nullopt;
    }
    BoundedByteReader<kParameterHeaderSize> header(span);
    uint16_t length = header.Load16<2>();
    if (length < kParameterHeaderSize || length > span.size()) {
      RTC_DLOG(LS_WARNING) << "Invalid parameter length field";
      return std::nullopt;
    }
    size_t length_with_padding = RoundUpTo4(length);
    if (length_with_padding > span.size()) {
      break;
    }
    span = span.subview(length_with_padding);
  }
  return Parameters(std::vector<uint8_t>(data.begin(), data.end()));
}
}  // namespace dcsctp
