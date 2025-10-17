/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 30, 2025.
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
#ifndef NET_DCSCTP_PACKET_ERROR_CAUSE_RESTART_OF_AN_ASSOCIATION_WITH_NEW_ADDRESS_CAUSE_H_
#define NET_DCSCTP_PACKET_ERROR_CAUSE_RESTART_OF_AN_ASSOCIATION_WITH_NEW_ADDRESS_CAUSE_H_
#include <stddef.h>

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "net/dcsctp/packet/error_cause/error_cause.h"
#include "net/dcsctp/packet/tlv_trait.h"

namespace dcsctp {

// https://tools.ietf.org/html/rfc4960#section-3.3.10.11
struct RestartOfAnAssociationWithNewAddressesCauseConfig
    : public ParameterConfig {
  static constexpr int kType = 11;
  static constexpr size_t kHeaderSize = 4;
  static constexpr size_t kVariableLengthAlignment = 1;
};

class RestartOfAnAssociationWithNewAddressesCause
    : public Parameter,
      public TLVTrait<RestartOfAnAssociationWithNewAddressesCauseConfig> {
 public:
  static constexpr int kType =
      RestartOfAnAssociationWithNewAddressesCauseConfig::kType;

  explicit RestartOfAnAssociationWithNewAddressesCause(
      rtc::ArrayView<const uint8_t> new_address_tlvs)
      : new_address_tlvs_(new_address_tlvs.begin(), new_address_tlvs.end()) {}

  static std::optional<RestartOfAnAssociationWithNewAddressesCause> Parse(
      rtc::ArrayView<const uint8_t> data);

  void SerializeTo(std::vector<uint8_t>& out) const override;
  std::string ToString() const override;

  rtc::ArrayView<const uint8_t> new_address_tlvs() const {
    return new_address_tlvs_;
  }

 private:
  std::vector<uint8_t> new_address_tlvs_;
};
}  // namespace dcsctp

#endif  // NET_DCSCTP_PACKET_ERROR_CAUSE_RESTART_OF_AN_ASSOCIATION_WITH_NEW_ADDRESS_CAUSE_H_
