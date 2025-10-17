/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 1, 2022.
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
#ifndef TEST_SCOPED_KEY_VALUE_CONFIG_H_
#define TEST_SCOPED_KEY_VALUE_CONFIG_H_

#include <functional>
#include <map>
#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "api/field_trials_registry.h"
#include "test/field_trial.h"

namespace webrtc {
namespace test {

class ScopedKeyValueConfig : public FieldTrialsRegistry {
 public:
  virtual ~ScopedKeyValueConfig();
  ScopedKeyValueConfig();
  explicit ScopedKeyValueConfig(absl::string_view s);
  ScopedKeyValueConfig(ScopedKeyValueConfig& parent, absl::string_view s);

 private:
  ScopedKeyValueConfig(ScopedKeyValueConfig* parent, absl::string_view s);
  ScopedKeyValueConfig* GetRoot(ScopedKeyValueConfig* n);
  std::string GetValue(absl::string_view key) const override;
  std::string LookupRecurse(absl::string_view key) const;

  ScopedKeyValueConfig* const parent_;

  // The leaf in a list of stacked ScopedKeyValueConfig.
  // Only set on root (e.g with parent_ == nullptr).
  const ScopedKeyValueConfig* leaf_;

  // Unlike std::less<std::string>, std::less<> is transparent and allows
  // heterogeneous lookup directly with absl::string_view.
  std::map<std::string, std::string, std::less<>> key_value_map_;
  std::unique_ptr<ScopedFieldTrials> scoped_field_trials_;
};

}  // namespace test
}  // namespace webrtc

#endif  // TEST_SCOPED_KEY_VALUE_CONFIG_H_
