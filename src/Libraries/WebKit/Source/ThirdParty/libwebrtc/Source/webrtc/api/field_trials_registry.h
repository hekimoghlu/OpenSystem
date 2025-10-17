/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 6, 2023.
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
#ifndef API_FIELD_TRIALS_REGISTRY_H_
#define API_FIELD_TRIALS_REGISTRY_H_

#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "api/field_trials_view.h"
#include "rtc_base/containers/flat_set.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// Abstract base class for a field trial registry that verifies that any looked
// up key has been pre-registered in accordance with `g3doc/field-trials.md`.
class RTC_EXPORT FieldTrialsRegistry : public FieldTrialsView {
 public:
  FieldTrialsRegistry() = default;

  FieldTrialsRegistry(const FieldTrialsRegistry&) = default;
  FieldTrialsRegistry& operator=(const FieldTrialsRegistry&) = default;

  ~FieldTrialsRegistry() override = default;

  // Verifies that `key` is a registered field trial and then returns the
  // configured value for `key` or an empty string if the field trial isn't
  // configured.
  std::string Lookup(absl::string_view key) const override;

  // Register additional `keys` for testing. This should only be used for
  // imaginary keys that are never used outside test code.
  void RegisterKeysForTesting(flat_set<std::string> keys) {
    test_keys_ = std::move(keys);
  }

 private:
  virtual std::string GetValue(absl::string_view key) const = 0;

  // Imaginary keys only used for testing.
  flat_set<std::string> test_keys_;
};

}  // namespace webrtc

#endif  // API_FIELD_TRIALS_REGISTRY_H_
