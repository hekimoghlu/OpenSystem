/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 3, 2023.
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
#include "rtc_base/experiments/field_trial_list.h"

#include "absl/strings/string_view.h"

namespace webrtc {

FieldTrialListBase::FieldTrialListBase(absl::string_view key)
    : FieldTrialParameterInterface(key),
      failed_(false),
      parse_got_called_(false) {}

bool FieldTrialListBase::Failed() const {
  return failed_;
}
bool FieldTrialListBase::Used() const {
  return parse_got_called_;
}

int FieldTrialListWrapper::Length() {
  return GetList()->Size();
}
bool FieldTrialListWrapper::Failed() {
  return GetList()->Failed();
}
bool FieldTrialListWrapper::Used() {
  return GetList()->Used();
}

bool FieldTrialStructListBase::Parse(std::optional<std::string> str_value) {
  RTC_DCHECK_NOTREACHED();
  return true;
}

int FieldTrialStructListBase::ValidateAndGetLength() {
  int length = -1;
  for (std::unique_ptr<FieldTrialListWrapper>& list : sub_lists_) {
    if (list->Failed())
      return -1;
    else if (!list->Used())
      continue;
    else if (length == -1)
      length = list->Length();
    else if (length != list->Length())
      return -1;
  }

  return length;
}

}  // namespace webrtc
