/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 15, 2022.
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
#ifndef NET_DCSCTP_PACKET_PARAMETER_OUTGOING_SSN_RESET_REQUEST_PARAMETER_H_
#define NET_DCSCTP_PACKET_PARAMETER_OUTGOING_SSN_RESET_REQUEST_PARAMETER_H_
#include <stddef.h>
#include <stdint.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "net/dcsctp/common/internal_types.h"
#include "net/dcsctp/packet/parameter/parameter.h"
#include "net/dcsctp/packet/tlv_trait.h"
#include "net/dcsctp/public/types.h"

namespace dcsctp {

// https://tools.ietf.org/html/rfc6525#section-4.1
struct OutgoingSSNResetRequestParameterConfig : ParameterConfig {
  static constexpr int kType = 13;
  static constexpr size_t kHeaderSize = 16;
  static constexpr size_t kVariableLengthAlignment = 2;
};

class OutgoingSSNResetRequestParameter
    : public Parameter,
      public TLVTrait<OutgoingSSNResetRequestParameterConfig> {
 public:
  static constexpr int kType = OutgoingSSNResetRequestParameterConfig::kType;

  explicit OutgoingSSNResetRequestParameter(
      ReconfigRequestSN request_sequence_number,
      ReconfigRequestSN response_sequence_number,
      TSN sender_last_assigned_tsn,
      std::vector<StreamID> stream_ids)
      : request_sequence_number_(request_sequence_number),
        response_sequence_number_(response_sequence_number),
        sender_last_assigned_tsn_(sender_last_assigned_tsn),
        stream_ids_(std::move(stream_ids)) {}

  static std::optional<OutgoingSSNResetRequestParameter> Parse(
      rtc::ArrayView<const uint8_t> data);

  void SerializeTo(std::vector<uint8_t>& out) const override;
  std::string ToString() const override;

  ReconfigRequestSN request_sequence_number() const {
    return request_sequence_number_;
  }
  ReconfigRequestSN response_sequence_number() const {
    return response_sequence_number_;
  }
  TSN sender_last_assigned_tsn() const { return sender_last_assigned_tsn_; }
  rtc::ArrayView<const StreamID> stream_ids() const { return stream_ids_; }

 private:
  static constexpr size_t kStreamIdSize = sizeof(uint16_t);

  ReconfigRequestSN request_sequence_number_;
  ReconfigRequestSN response_sequence_number_;
  TSN sender_last_assigned_tsn_;
  std::vector<StreamID> stream_ids_;
};

}  // namespace dcsctp

#endif  // NET_DCSCTP_PACKET_PARAMETER_OUTGOING_SSN_RESET_REQUEST_PARAMETER_H_
