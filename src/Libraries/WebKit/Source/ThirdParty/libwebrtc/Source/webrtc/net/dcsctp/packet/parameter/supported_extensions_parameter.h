/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 15, 2024.
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
#ifndef NET_DCSCTP_PACKET_PARAMETER_SUPPORTED_EXTENSIONS_PARAMETER_H_
#define NET_DCSCTP_PACKET_PARAMETER_SUPPORTED_EXTENSIONS_PARAMETER_H_
#include <stddef.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "net/dcsctp/packet/parameter/parameter.h"
#include "net/dcsctp/packet/tlv_trait.h"

namespace dcsctp {

// https://tools.ietf.org/html/rfc5061#section-4.2.7
struct SupportedExtensionsParameterConfig : ParameterConfig {
  static constexpr int kType = 0x8008;
  static constexpr size_t kHeaderSize = 4;
  static constexpr size_t kVariableLengthAlignment = 1;
};

class SupportedExtensionsParameter
    : public Parameter,
      public TLVTrait<SupportedExtensionsParameterConfig> {
 public:
  static constexpr int kType = SupportedExtensionsParameterConfig::kType;

  explicit SupportedExtensionsParameter(std::vector<uint8_t> chunk_types)
      : chunk_types_(std::move(chunk_types)) {}

  static std::optional<SupportedExtensionsParameter> Parse(
      rtc::ArrayView<const uint8_t> data);

  void SerializeTo(std::vector<uint8_t>& out) const override;
  std::string ToString() const override;

  bool supports(uint8_t chunk_type) const {
    return std::find(chunk_types_.begin(), chunk_types_.end(), chunk_type) !=
           chunk_types_.end();
  }

  rtc::ArrayView<const uint8_t> chunk_types() const { return chunk_types_; }

 private:
  std::vector<uint8_t> chunk_types_;
};
}  // namespace dcsctp

#endif  // NET_DCSCTP_PACKET_PARAMETER_SUPPORTED_EXTENSIONS_PARAMETER_H_
