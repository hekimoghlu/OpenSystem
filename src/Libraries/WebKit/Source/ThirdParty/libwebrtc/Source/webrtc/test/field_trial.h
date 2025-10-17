/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 13, 2022.
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
#ifndef TEST_FIELD_TRIAL_H_
#define TEST_FIELD_TRIAL_H_

#include <string>

#include "absl/strings/string_view.h"

namespace webrtc {
namespace test {

// This class is used to override field-trial configs within specific tests.
// After this class goes out of scope previous field trials will be restored.
class ScopedFieldTrials {
 public:
  explicit ScopedFieldTrials(absl::string_view config);
  ScopedFieldTrials(const ScopedFieldTrials&) = delete;
  ScopedFieldTrials& operator=(const ScopedFieldTrials&) = delete;
  ~ScopedFieldTrials();

 private:
  std::string current_field_trials_;
  const char* previous_field_trials_;
};

}  // namespace test
}  // namespace webrtc

#endif  // TEST_FIELD_TRIAL_H_
