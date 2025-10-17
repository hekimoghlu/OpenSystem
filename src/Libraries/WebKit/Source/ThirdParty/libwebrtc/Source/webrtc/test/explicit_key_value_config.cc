/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 1, 2025.
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
#include "test/explicit_key_value_config.h"

#include "absl/strings/string_view.h"
#include "rtc_base/checks.h"

namespace webrtc {
namespace test {

ExplicitKeyValueConfig::ExplicitKeyValueConfig(absl::string_view s) {
  std::string::size_type field_start = 0;
  while (field_start < s.size()) {
    std::string::size_type separator_pos = s.find('/', field_start);
    RTC_CHECK_NE(separator_pos, std::string::npos)
        << "Missing separator '/' after field trial key.";
    RTC_CHECK_GT(separator_pos, field_start)
        << "Field trial key cannot be empty.";
    std::string key(s.substr(field_start, separator_pos - field_start));
    field_start = separator_pos + 1;

    RTC_CHECK_LT(field_start, s.size())
        << "Missing value after field trial key. String ended.";
    separator_pos = s.find('/', field_start);
    RTC_CHECK_NE(separator_pos, std::string::npos)
        << "Missing terminating '/' in field trial string.";
    RTC_CHECK_GT(separator_pos, field_start)
        << "Field trial value cannot be empty.";
    std::string value(s.substr(field_start, separator_pos - field_start));
    field_start = separator_pos + 1;

    key_value_map_[key] = value;
  }
  // This check is technically redundant due to earlier checks.
  // We nevertheless keep the check to make it clear that the entire
  // string has been processed, and without indexing past the end.
  RTC_CHECK_EQ(field_start, s.size());
}

std::string ExplicitKeyValueConfig::GetValue(absl::string_view key) const {
  auto it = key_value_map_.find(key);
  if (it != key_value_map_.end())
    return it->second;
  return "";
}

}  // namespace test
}  // namespace webrtc
