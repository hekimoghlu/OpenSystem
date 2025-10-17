/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 6, 2025.
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
#include "test/scoped_key_value_config.h"

#include "rtc_base/checks.h"
#include "system_wrappers/include/field_trial.h"
#include "test/field_trial.h"

namespace {

// This part is copied from system_wrappers/field_trial.cc.
void InsertIntoMap(
    std::map<std::string, std::string, std::less<>>& key_value_map,
    absl::string_view s) {
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

    key_value_map[key] = value;
  }
  // This check is technically redundant due to earlier checks.
  // We nevertheless keep the check to make it clear that the entire
  // string has been processed, and without indexing past the end.
  RTC_CHECK_EQ(field_start, s.size());
}

}  // namespace

namespace webrtc {
namespace test {

ScopedKeyValueConfig::ScopedKeyValueConfig()
    : ScopedKeyValueConfig(nullptr, "") {}

ScopedKeyValueConfig::ScopedKeyValueConfig(absl::string_view s)
    : ScopedKeyValueConfig(nullptr, s) {}

ScopedKeyValueConfig::ScopedKeyValueConfig(ScopedKeyValueConfig& parent,
                                           absl::string_view s)
    : ScopedKeyValueConfig(&parent, s) {}

ScopedKeyValueConfig::ScopedKeyValueConfig(ScopedKeyValueConfig* parent,
                                           absl::string_view s)
    : parent_(parent), leaf_(nullptr) {
  InsertIntoMap(key_value_map_, s);

  if (!s.empty()) {
    // Also store field trials in global string (until we get rid of it).
    scoped_field_trials_ = std::make_unique<ScopedFieldTrials>(s);
  }

  if (parent == nullptr) {
    // We are root, set leaf_.
    leaf_ = this;
  } else {
    // Link root to new leaf.
    GetRoot(parent)->leaf_ = this;
    RTC_DCHECK(leaf_ == nullptr);
  }
}

ScopedKeyValueConfig::~ScopedKeyValueConfig() {
  if (parent_) {
    GetRoot(parent_)->leaf_ = parent_;
  }
}

ScopedKeyValueConfig* ScopedKeyValueConfig::GetRoot(ScopedKeyValueConfig* n) {
  while (n->parent_ != nullptr) {
    n = n->parent_;
  }
  return n;
}

std::string ScopedKeyValueConfig::GetValue(absl::string_view key) const {
  if (parent_ == nullptr) {
    return leaf_->LookupRecurse(key);
  } else {
    return LookupRecurse(key);
  }
}

std::string ScopedKeyValueConfig::LookupRecurse(absl::string_view key) const {
  auto it = key_value_map_.find(key);
  if (it != key_value_map_.end())
    return it->second;

  if (parent_) {
    return parent_->LookupRecurse(key);
  }

  // When at the root, check the global string so that test programs using
  // a mix between ScopedKeyValueConfig and the global string continue to work
  return webrtc::field_trial::FindFullName(std::string(key));
}

}  // namespace test
}  // namespace webrtc
