/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 26, 2025.
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
#include "test/field_trial.h"

#include <string>

#include "absl/strings/string_view.h"
#include "rtc_base/checks.h"
#include "system_wrappers/include/field_trial.h"

namespace webrtc {
namespace test {

ScopedFieldTrials::ScopedFieldTrials(absl::string_view config)
    : current_field_trials_(config),
      previous_field_trials_(webrtc::field_trial::GetFieldTrialString()) {
  RTC_CHECK(webrtc::field_trial::FieldTrialsStringIsValid(
      current_field_trials_.c_str()))
      << "Invalid field trials string: " << current_field_trials_;
  webrtc::field_trial::InitFieldTrialsFromString(current_field_trials_.c_str());
}

ScopedFieldTrials::~ScopedFieldTrials() {
  RTC_CHECK(
      webrtc::field_trial::FieldTrialsStringIsValid(previous_field_trials_))
      << "Invalid field trials string: " << previous_field_trials_;
  webrtc::field_trial::InitFieldTrialsFromString(previous_field_trials_);
}

}  // namespace test
}  // namespace webrtc
