/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 29, 2023.
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
#ifndef API_FIELD_TRIALS_VIEW_H_
#define API_FIELD_TRIALS_VIEW_H_

#include <string>

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// An interface that provides the means to access field trials.
//
// Note that there are no guarantess that the meaning of a particular key-value
// mapping will be preserved over time and no announcements will be made if they
// are changed. It's up to the library user to ensure that the behavior does not
// break.
class RTC_EXPORT FieldTrialsView {
 public:
  virtual ~FieldTrialsView() = default;

  // Returns the configured value for `key` or an empty string if the field
  // trial isn't configured.
  virtual std::string Lookup(absl::string_view key) const = 0;

  bool IsEnabled(absl::string_view key) const {
    return absl::StartsWith(Lookup(key), "Enabled");
  }

  bool IsDisabled(absl::string_view key) const {
    return absl::StartsWith(Lookup(key), "Disabled");
  }
};

}  // namespace webrtc

#endif  // API_FIELD_TRIALS_VIEW_H_
