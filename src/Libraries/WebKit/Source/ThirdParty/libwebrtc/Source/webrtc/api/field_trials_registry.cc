/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 26, 2025.
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
#include "api/field_trials_registry.h"

#include <string>

#include "absl/strings/string_view.h"
// IWYU pragma: begin_keep
#include "absl/algorithm/container.h"
#if !defined(WEBRTC_WEBKIT_BUILD)
#include "experiments/registered_field_trials.h"
#endif
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"
// IWYU pragma: end_keep

namespace webrtc {

std::string FieldTrialsRegistry::Lookup(absl::string_view key) const {
#if WEBRTC_STRICT_FIELD_TRIALS == 1
  RTC_DCHECK(absl::c_linear_search(kRegisteredFieldTrials, key) ||
             test_keys_.contains(key))
      << key << " is not registered, see g3doc/field-trials.md.";
#elif WEBRTC_STRICT_FIELD_TRIALS == 2
  RTC_LOG_IF(LS_WARNING, !(absl::c_linear_search(kRegisteredFieldTrials, key) ||
                           test_keys_.contains(key)))
      << key << " is not registered, see g3doc/field-trials.md.";
#endif
  return GetValue(key);
}

}  // namespace webrtc
