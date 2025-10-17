/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 25, 2023.
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
#include "rtc_base/string_to_number.h"

#include <ctype.h>

#include <cerrno>
#include <cstdlib>

#include "rtc_base/checks.h"

namespace rtc {
namespace string_to_number_internal {

std::optional<signed_type> ParseSigned(absl::string_view str, int base) {
  if (str.empty())
    return std::nullopt;

  if (isdigit(static_cast<unsigned char>(str[0])) || str[0] == '-') {
    std::string str_str(str);
    char* end = nullptr;
    errno = 0;
    const signed_type value = std::strtoll(str_str.c_str(), &end, base);
    // Check for errors and also make sure that there were no embedded nuls in
    // the input string.
    if (end == str_str.c_str() + str_str.size() && errno == 0) {
      return value;
    }
  }
  return std::nullopt;
}

std::optional<unsigned_type> ParseUnsigned(absl::string_view str, int base) {
  if (str.empty())
    return std::nullopt;

  if (isdigit(static_cast<unsigned char>(str[0])) || str[0] == '-') {
    std::string str_str(str);
    // Explicitly discard negative values. std::strtoull parsing causes unsigned
    // wraparound. We cannot just reject values that start with -, though, since
    // -0 is perfectly fine, as is -0000000000000000000000000000000.
    const bool is_negative = str[0] == '-';
    char* end = nullptr;
    errno = 0;
    const unsigned_type value = std::strtoull(str_str.c_str(), &end, base);
    // Check for errors and also make sure that there were no embedded nuls in
    // the input string.
    if (end == str_str.c_str() + str_str.size() && errno == 0 &&
        (value == 0 || !is_negative)) {
      return value;
    }
  }
  return std::nullopt;
}

template <typename T>
T StrToT(const char* str, char** str_end);

template <>
inline float StrToT(const char* str, char** str_end) {
  return std::strtof(str, str_end);
}

template <>
inline double StrToT(const char* str, char** str_end) {
  return std::strtod(str, str_end);
}

template <>
inline long double StrToT(const char* str, char** str_end) {
  return std::strtold(str, str_end);
}

template <typename T>
std::optional<T> ParseFloatingPoint(absl::string_view str) {
  if (str.empty())
    return std::nullopt;

  if (str[0] == '\0')
    return std::nullopt;
  std::string str_str(str);
  char* end = nullptr;
  errno = 0;
  const T value = StrToT<T>(str_str.c_str(), &end);
  if (end == str_str.c_str() + str_str.size() && errno == 0) {
    return value;
  }
  return std::nullopt;
}

template std::optional<float> ParseFloatingPoint(absl::string_view str);
template std::optional<double> ParseFloatingPoint(absl::string_view str);
template std::optional<long double> ParseFloatingPoint(absl::string_view str);

}  // namespace string_to_number_internal
}  // namespace rtc
