/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 29, 2021.
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
#ifndef NET_DCSCTP_PACKET_ERROR_CAUSE_NO_USER_DATA_CAUSE_H_
#define NET_DCSCTP_PACKET_ERROR_CAUSE_NO_USER_DATA_CAUSE_H_
#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "net/dcsctp/common/internal_types.h"
#include "net/dcsctp/packet/error_cause/error_cause.h"
#include "net/dcsctp/packet/tlv_trait.h"
namespace dcsctp {

// https://tools.ietf.org/html/rfc4960#section-3.3.10.9
struct NoUserDataCauseConfig : public ParameterConfig {
  static constexpr int kType = 9;
  static constexpr size_t kHeaderSize = 8;
  static constexpr size_t kVariableLengthAlignment = 0;
};

class NoUserDataCause : public Parameter,
                        public TLVTrait<NoUserDataCauseConfig> {
 public:
  static constexpr int kType = NoUserDataCauseConfig::kType;

  explicit NoUserDataCause(TSN tsn) : tsn_(tsn) {}

  static std::optional<NoUserDataCause> Parse(
      rtc::ArrayView<const uint8_t> data);

  void SerializeTo(std::vector<uint8_t>& out) const override;
  std::string ToString() const override;

  TSN tsn() const { return tsn_; }

 private:
  TSN tsn_;
};

}  // namespace dcsctp

#endif  // NET_DCSCTP_PACKET_ERROR_CAUSE_NO_USER_DATA_CAUSE_H_
