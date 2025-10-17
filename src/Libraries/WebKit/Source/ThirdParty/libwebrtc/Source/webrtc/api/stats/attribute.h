/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 22, 2024.
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
#ifndef API_STATS_ATTRIBUTE_H_
#define API_STATS_ATTRIBUTE_H_

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/types/variant.h"
#include "rtc_base/checks.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// A light-weight wrapper of an RTCStats attribute, i.e. an individual metric of
// type std::optional<T>.
class RTC_EXPORT Attribute {
 public:
  // All supported attribute types.
  typedef absl::variant<const std::optional<bool>*,
                        const std::optional<int32_t>*,
                        const std::optional<uint32_t>*,
                        const std::optional<int64_t>*,
                        const std::optional<uint64_t>*,
                        const std::optional<double>*,
                        const std::optional<std::string>*,
                        const std::optional<std::vector<bool>>*,
                        const std::optional<std::vector<int32_t>>*,
                        const std::optional<std::vector<uint32_t>>*,
                        const std::optional<std::vector<int64_t>>*,
                        const std::optional<std::vector<uint64_t>>*,
                        const std::optional<std::vector<double>>*,
                        const std::optional<std::vector<std::string>>*,
                        const std::optional<std::map<std::string, uint64_t>>*,
                        const std::optional<std::map<std::string, double>>*>
      StatVariant;

  template <typename T>
  Attribute(const char* name, const std::optional<T>* attribute)
      : name_(name), attribute_(attribute) {}

  const char* name() const;
  const StatVariant& as_variant() const;

  bool has_value() const;
  template <typename T>
  bool holds_alternative() const {
    return absl::holds_alternative<const std::optional<T>*>(attribute_);
  }
  template <typename T>
  const std::optional<T>& as_optional() const {
    RTC_CHECK(holds_alternative<T>());
    return *absl::get<const std::optional<T>*>(attribute_);
  }
  template <typename T>
  const T& get() const {
    RTC_CHECK(holds_alternative<T>());
    RTC_CHECK(has_value());
    return absl::get<const std::optional<T>*>(attribute_)->value();
  }

  bool is_sequence() const;
  bool is_string() const;
  std::string ToString() const;

  bool operator==(const Attribute& other) const;
  bool operator!=(const Attribute& other) const;

 private:
  const char* name_;
  StatVariant attribute_;
};

struct RTC_EXPORT AttributeInit {
  AttributeInit(const char* name, const Attribute::StatVariant& variant);

  const char* name;
  Attribute::StatVariant variant;
};

}  // namespace webrtc

#endif  // API_STATS_ATTRIBUTE_H_
