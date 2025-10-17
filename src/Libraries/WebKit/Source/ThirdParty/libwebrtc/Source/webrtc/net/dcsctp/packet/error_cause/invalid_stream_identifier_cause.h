/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 2, 2023.
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
#ifndef NET_DCSCTP_PACKET_ERROR_CAUSE_INVALID_STREAM_IDENTIFIER_CAUSE_H_
#define NET_DCSCTP_PACKET_ERROR_CAUSE_INVALID_STREAM_IDENTIFIER_CAUSE_H_
#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "net/dcsctp/packet/error_cause/error_cause.h"
#include "net/dcsctp/packet/tlv_trait.h"
#include "net/dcsctp/public/types.h"

namespace dcsctp {

// https://tools.ietf.org/html/rfc4960#section-3.3.10.1
struct InvalidStreamIdentifierCauseConfig : public ParameterConfig {
  static constexpr int kType = 1;
  static constexpr size_t kHeaderSize = 8;
  static constexpr size_t kVariableLengthAlignment = 0;
};

class InvalidStreamIdentifierCause
    : public Parameter,
      public TLVTrait<InvalidStreamIdentifierCauseConfig> {
 public:
  static constexpr int kType = InvalidStreamIdentifierCauseConfig::kType;

  explicit InvalidStreamIdentifierCause(StreamID stream_id)
      : stream_id_(stream_id) {}

  static std::optional<InvalidStreamIdentifierCause> Parse(
      rtc::ArrayView<const uint8_t> data);

  void SerializeTo(std::vector<uint8_t>& out) const override;
  std::string ToString() const override;

  StreamID stream_id() const { return stream_id_; }

 private:
  StreamID stream_id_;
};

}  // namespace dcsctp

#endif  // NET_DCSCTP_PACKET_ERROR_CAUSE_INVALID_STREAM_IDENTIFIER_CAUSE_H_
