/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 5, 2021.
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
#ifndef NET_DCSCTP_PACKET_PARAMETER_PARAMETER_H_
#define NET_DCSCTP_PACKET_PARAMETER_PARAMETER_H_

#include <stddef.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "net/dcsctp/packet/tlv_trait.h"
#include "rtc_base/strings/string_builder.h"

namespace dcsctp {

class Parameter {
 public:
  Parameter() {}
  virtual ~Parameter() = default;

  Parameter(const Parameter& other) = default;
  Parameter& operator=(const Parameter& other) = default;

  virtual void SerializeTo(std::vector<uint8_t>& out) const = 0;
  virtual std::string ToString() const = 0;
};

struct ParameterDescriptor {
  ParameterDescriptor(uint16_t type, rtc::ArrayView<const uint8_t> data)
      : type(type), data(data) {}
  uint16_t type;
  rtc::ArrayView<const uint8_t> data;
};

class Parameters {
 public:
  class Builder {
   public:
    Builder() {}
    Builder& Add(const Parameter& p);
    Parameters Build() { return Parameters(std::move(data_)); }

   private:
    std::vector<uint8_t> data_;
  };

  static std::optional<Parameters> Parse(rtc::ArrayView<const uint8_t> data);

  Parameters() {}
  Parameters(Parameters&& other) = default;
  Parameters& operator=(Parameters&& other) = default;

  rtc::ArrayView<const uint8_t> data() const { return data_; }
  std::vector<ParameterDescriptor> descriptors() const;

  template <typename P>
  std::optional<P> get() const {
    static_assert(std::is_base_of<Parameter, P>::value,
                  "Template parameter not derived from Parameter");
    for (const auto& p : descriptors()) {
      if (p.type == P::kType) {
        return P::Parse(p.data);
      }
    }
    return std::nullopt;
  }

 private:
  explicit Parameters(std::vector<uint8_t> data) : data_(std::move(data)) {}
  std::vector<uint8_t> data_;
};

struct ParameterConfig {
  static constexpr int kTypeSizeInBytes = 2;
};

}  // namespace dcsctp

#endif  // NET_DCSCTP_PACKET_PARAMETER_PARAMETER_H_
