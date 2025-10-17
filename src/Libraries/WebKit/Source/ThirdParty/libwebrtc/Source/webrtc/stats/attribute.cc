/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 6, 2023.
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
#include "api/stats/attribute.h"

#include <string>

#include "absl/types/variant.h"
#include "rtc_base/arraysize.h"
#include "rtc_base/checks.h"
#include "rtc_base/string_encode.h"
#include "rtc_base/strings/string_builder.h"

namespace webrtc {

namespace {

struct VisitIsSequence {
  // Any type of vector is a sequence.
  template <typename T>
  bool operator()(const std::optional<std::vector<T>>* attribute) {
    return true;
  }
  // Any other type is not.
  template <typename T>
  bool operator()(const std::optional<T>* attribute) {
    return false;
  }
};

// Converts the attribute to string in a JSON-compatible way.
struct VisitToString {
  template <typename T,
            typename std::enable_if_t<
                std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
                    std::is_same_v<T, bool> || std::is_same_v<T, std::string>,
                bool> = true>
  std::string ValueToString(const T& value) {
    return rtc::ToString(value);
  }
  // Convert 64-bit integers to doubles before converting to string because JSON
  // represents all numbers as floating points with ~15 digits of precision.
  template <typename T,
            typename std::enable_if_t<std::is_same_v<T, int64_t> ||
                                          std::is_same_v<T, uint64_t> ||
                                          std::is_same_v<T, double>,
                                      bool> = true>
  std::string ValueToString(const T& value) {
    char buf[32];
    const int len = std::snprintf(&buf[0], arraysize(buf), "%.16g",
                                  static_cast<double>(value));
    RTC_DCHECK_LE(len, arraysize(buf));
    return std::string(&buf[0], len);
  }

  // Vector attributes.
  template <typename T>
  std::string operator()(const std::optional<std::vector<T>>* attribute) {
    rtc::StringBuilder sb;
    sb << "[";
    const char* separator = "";
    constexpr bool element_is_string = std::is_same<T, std::string>::value;
    for (const T& element : attribute->value()) {
      sb << separator;
      if (element_is_string) {
        sb << "\"";
      }
      sb << ValueToString(element);
      if (element_is_string) {
        sb << "\"";
      }
      separator = ",";
    }
    sb << "]";
    return sb.Release();
  }
  // Map attributes.
  template <typename T>
  std::string operator()(
      const std::optional<std::map<std::string, T>>* attribute) {
    rtc::StringBuilder sb;
    sb << "{";
    const char* separator = "";
    constexpr bool element_is_string = std::is_same<T, std::string>::value;
    for (const auto& pair : attribute->value()) {
      sb << separator;
      sb << "\"" << pair.first << "\":";
      if (element_is_string) {
        sb << "\"";
      }
      sb << ValueToString(pair.second);
      if (element_is_string) {
        sb << "\"";
      }
      separator = ",";
    }
    sb << "}";
    return sb.Release();
  }
  // Simple attributes.
  template <typename T>
  std::string operator()(const std::optional<T>* attribute) {
    return ValueToString(attribute->value());
  }
};

struct VisitIsEqual {
  template <typename T>
  bool operator()(const std::optional<T>* attribute) {
    if (!other.holds_alternative<T>()) {
      return false;
    }
    return *attribute == other.as_optional<T>();
  }

  const Attribute& other;
};

}  // namespace

const char* Attribute::name() const {
  return name_;
}

const Attribute::StatVariant& Attribute::as_variant() const {
  return attribute_;
}

bool Attribute::has_value() const {
  return absl::visit([](const auto* attr) { return attr->has_value(); },
                     attribute_);
}

bool Attribute::is_sequence() const {
  return absl::visit(VisitIsSequence(), attribute_);
}

bool Attribute::is_string() const {
  return absl::holds_alternative<const std::optional<std::string>*>(attribute_);
}

std::string Attribute::ToString() const {
  if (!has_value()) {
    return "null";
  }
  return absl::visit(VisitToString(), attribute_);
}

bool Attribute::operator==(const Attribute& other) const {
  return absl::visit(VisitIsEqual{.other = other}, attribute_);
}

bool Attribute::operator!=(const Attribute& other) const {
  return !(*this == other);
}

AttributeInit::AttributeInit(const char* name,
                             const Attribute::StatVariant& variant)
    : name(name), variant(variant) {}

}  // namespace webrtc
