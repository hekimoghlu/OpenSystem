/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 17, 2023.
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
#ifndef RTC_BASE_STRINGS_STR_JOIN_H_
#define RTC_BASE_STRINGS_STR_JOIN_H_

#include <string>

#include "absl/strings/string_view.h"
#include "rtc_base/strings/string_builder.h"

namespace webrtc {

template <typename Range>
std::string StrJoin(const Range& seq, absl::string_view delimiter) {
  rtc::StringBuilder sb;
  int idx = 0;

  for (const typename Range::value_type& elem : seq) {
    if (idx > 0) {
      sb << delimiter;
    }
    sb << elem;

    ++idx;
  }
  return sb.Release();
}

template <typename Range, typename Functor>
std::string StrJoin(const Range& seq,
                    absl::string_view delimiter,
                    const Functor& fn) {
  rtc::StringBuilder sb;
  int idx = 0;

  for (const typename Range::value_type& elem : seq) {
    if (idx > 0) {
      sb << delimiter;
    }
    fn(sb, elem);

    ++idx;
  }
  return sb.Release();
}

}  // namespace webrtc

#endif  // RTC_BASE_STRINGS_STR_JOIN_H_
