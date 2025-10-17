/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 12, 2021.
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
#ifndef API_FIELD_TRIALS_H_
#define API_FIELD_TRIALS_H_

#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "api/field_trials_registry.h"
#include "rtc_base/containers/flat_map.h"

namespace webrtc {

// The FieldTrials class is used to inject field trials into webrtc.
//
// Field trials allow webrtc clients (such as Chromium) to turn on feature code
// in binaries out in the field and gather information with that.
//
// They are designed to be easy to use with Chromium field trials and to speed
// up developers by reducing the need to wire up APIs to control whether a
// feature is on/off.
//
// The field trials are injected into objects that use them at creation time.
//
// NOTE: Creating multiple FieldTrials-object is currently prohibited
// until we remove the global string (TODO(bugs.webrtc.org/10335))
// (unless using CreateNoGlobal):
class FieldTrials : public FieldTrialsRegistry {
 public:
  explicit FieldTrials(const std::string& s);
  ~FieldTrials();

  // Create a FieldTrials object that is not reading/writing from
  // global variable (i.e can not be used for all parts of webrtc).
  static std::unique_ptr<FieldTrials> CreateNoGlobal(const std::string& s);

 private:
  explicit FieldTrials(const std::string& s, bool);

  std::string GetValue(absl::string_view key) const override;

  const bool uses_global_;
  const std::string field_trial_string_;
  const char* const previous_field_trial_string_;
  const flat_map<std::string, std::string> key_value_map_;
};

}  // namespace webrtc

#endif  // API_FIELD_TRIALS_H_
