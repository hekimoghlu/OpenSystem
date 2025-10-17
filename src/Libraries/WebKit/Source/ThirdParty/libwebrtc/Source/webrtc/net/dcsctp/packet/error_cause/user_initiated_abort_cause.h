/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 28, 2023.
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
#ifndef NET_DCSCTP_PACKET_ERROR_CAUSE_USER_INITIATED_ABORT_CAUSE_H_
#define NET_DCSCTP_PACKET_ERROR_CAUSE_USER_INITIATED_ABORT_CAUSE_H_
#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "net/dcsctp/packet/error_cause/error_cause.h"
#include "net/dcsctp/packet/tlv_trait.h"

namespace dcsctp {

// https://tools.ietf.org/html/rfc4960#section-3.3.10.12
struct UserInitiatedAbortCauseConfig : public ParameterConfig {
  static constexpr int kType = 12;
  static constexpr size_t kHeaderSize = 4;
  static constexpr size_t kVariableLengthAlignment = 1;
};

class UserInitiatedAbortCause : public Parameter,
                                public TLVTrait<UserInitiatedAbortCauseConfig> {
 public:
  static constexpr int kType = UserInitiatedAbortCauseConfig::kType;

  explicit UserInitiatedAbortCause(absl::string_view upper_layer_abort_reason)
      : upper_layer_abort_reason_(upper_layer_abort_reason) {}

  static std::optional<UserInitiatedAbortCause> Parse(
      rtc::ArrayView<const uint8_t> data);

  void SerializeTo(std::vector<uint8_t>& out) const override;
  std::string ToString() const override;

  absl::string_view upper_layer_abort_reason() const {
    return upper_layer_abort_reason_;
  }

 private:
  std::string upper_layer_abort_reason_;
};

}  // namespace dcsctp

#endif  // NET_DCSCTP_PACKET_ERROR_CAUSE_USER_INITIATED_ABORT_CAUSE_H_
