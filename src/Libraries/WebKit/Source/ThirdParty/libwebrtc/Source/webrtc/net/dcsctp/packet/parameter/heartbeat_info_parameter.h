/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 7, 2025.
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
#ifndef NET_DCSCTP_PACKET_PARAMETER_HEARTBEAT_INFO_PARAMETER_H_
#define NET_DCSCTP_PACKET_PARAMETER_HEARTBEAT_INFO_PARAMETER_H_
#include <stddef.h>

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "net/dcsctp/packet/parameter/parameter.h"
#include "net/dcsctp/packet/tlv_trait.h"

namespace dcsctp {

// https://tools.ietf.org/html/rfc4960#section-3.3.5
struct HeartbeatInfoParameterConfig : ParameterConfig {
  static constexpr int kType = 1;
  static constexpr size_t kHeaderSize = 4;
  static constexpr size_t kVariableLengthAlignment = 1;
};

class HeartbeatInfoParameter : public Parameter,
                               public TLVTrait<HeartbeatInfoParameterConfig> {
 public:
  static constexpr int kType = HeartbeatInfoParameterConfig::kType;

  explicit HeartbeatInfoParameter(rtc::ArrayView<const uint8_t> info)
      : info_(info.begin(), info.end()) {}

  static std::optional<HeartbeatInfoParameter> Parse(
      rtc::ArrayView<const uint8_t> data);

  void SerializeTo(std::vector<uint8_t>& out) const override;
  std::string ToString() const override;

  rtc::ArrayView<const uint8_t> info() const { return info_; }

 private:
  std::vector<uint8_t> info_;
};

}  // namespace dcsctp

#endif  // NET_DCSCTP_PACKET_PARAMETER_HEARTBEAT_INFO_PARAMETER_H_
