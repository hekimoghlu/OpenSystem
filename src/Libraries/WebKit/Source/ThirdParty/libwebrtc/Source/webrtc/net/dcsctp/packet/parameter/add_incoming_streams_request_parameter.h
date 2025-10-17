/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 6, 2024.
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
#ifndef NET_DCSCTP_PACKET_PARAMETER_ADD_INCOMING_STREAMS_REQUEST_PARAMETER_H_
#define NET_DCSCTP_PACKET_PARAMETER_ADD_INCOMING_STREAMS_REQUEST_PARAMETER_H_
#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "net/dcsctp/common/internal_types.h"
#include "net/dcsctp/packet/parameter/parameter.h"
#include "net/dcsctp/packet/tlv_trait.h"

namespace dcsctp {

// https://tools.ietf.org/html/rfc6525#section-4.6
struct AddIncomingStreamsRequestParameterConfig : ParameterConfig {
  static constexpr int kType = 18;
  static constexpr size_t kHeaderSize = 12;
  static constexpr size_t kVariableLengthAlignment = 0;
};

class AddIncomingStreamsRequestParameter
    : public Parameter,
      public TLVTrait<AddIncomingStreamsRequestParameterConfig> {
 public:
  static constexpr int kType = AddIncomingStreamsRequestParameterConfig::kType;

  explicit AddIncomingStreamsRequestParameter(
      ReconfigRequestSN request_sequence_number,
      uint16_t nbr_of_new_streams)
      : request_sequence_number_(request_sequence_number),
        nbr_of_new_streams_(nbr_of_new_streams) {}

  static std::optional<AddIncomingStreamsRequestParameter> Parse(
      rtc::ArrayView<const uint8_t> data);

  void SerializeTo(std::vector<uint8_t>& out) const override;
  std::string ToString() const override;

  ReconfigRequestSN request_sequence_number() const {
    return request_sequence_number_;
  }
  uint16_t nbr_of_new_streams() const { return nbr_of_new_streams_; }

 private:
  ReconfigRequestSN request_sequence_number_;
  uint16_t nbr_of_new_streams_;
};

}  // namespace dcsctp

#endif  // NET_DCSCTP_PACKET_PARAMETER_ADD_INCOMING_STREAMS_REQUEST_PARAMETER_H_
