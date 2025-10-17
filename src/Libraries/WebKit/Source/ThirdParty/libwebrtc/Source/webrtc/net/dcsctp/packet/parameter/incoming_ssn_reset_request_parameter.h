/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 10, 2022.
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
#ifndef NET_DCSCTP_PACKET_PARAMETER_INCOMING_SSN_RESET_REQUEST_PARAMETER_H_
#define NET_DCSCTP_PACKET_PARAMETER_INCOMING_SSN_RESET_REQUEST_PARAMETER_H_
#include <stddef.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "net/dcsctp/common/internal_types.h"
#include "net/dcsctp/packet/parameter/parameter.h"
#include "net/dcsctp/public/types.h"

namespace dcsctp {

// https://tools.ietf.org/html/rfc6525#section-4.2
struct IncomingSSNResetRequestParameterConfig : ParameterConfig {
  static constexpr int kType = 14;
  static constexpr size_t kHeaderSize = 8;
  static constexpr size_t kVariableLengthAlignment = 2;
};

class IncomingSSNResetRequestParameter
    : public Parameter,
      public TLVTrait<IncomingSSNResetRequestParameterConfig> {
 public:
  static constexpr int kType = IncomingSSNResetRequestParameterConfig::kType;

  explicit IncomingSSNResetRequestParameter(
      ReconfigRequestSN request_sequence_number,
      std::vector<StreamID> stream_ids)
      : request_sequence_number_(request_sequence_number),
        stream_ids_(std::move(stream_ids)) {}

  static std::optional<IncomingSSNResetRequestParameter> Parse(
      rtc::ArrayView<const uint8_t> data);

  void SerializeTo(std::vector<uint8_t>& out) const override;
  std::string ToString() const override;

  ReconfigRequestSN request_sequence_number() const {
    return request_sequence_number_;
  }
  rtc::ArrayView<const StreamID> stream_ids() const { return stream_ids_; }

 private:
  static constexpr size_t kStreamIdSize = sizeof(uint16_t);

  ReconfigRequestSN request_sequence_number_;
  std::vector<StreamID> stream_ids_;
};

}  // namespace dcsctp

#endif  // NET_DCSCTP_PACKET_PARAMETER_INCOMING_SSN_RESET_REQUEST_PARAMETER_H_
