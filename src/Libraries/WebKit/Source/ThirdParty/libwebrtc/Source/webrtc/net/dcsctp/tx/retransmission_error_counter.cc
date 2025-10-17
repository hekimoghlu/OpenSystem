/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 15, 2024.
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
#include "net/dcsctp/tx/retransmission_error_counter.h"

#include "absl/strings/string_view.h"
#include "rtc_base/logging.h"

namespace dcsctp {
bool RetransmissionErrorCounter::Increment(absl::string_view reason) {
  ++counter_;
  if (limit_.has_value() && counter_ > limit_.value()) {
    RTC_DLOG(LS_INFO) << log_prefix_ << reason
                      << ", too many retransmissions, counter=" << counter_;
    return false;
  }

  RTC_DLOG(LS_VERBOSE) << log_prefix_ << reason << ", new counter=" << counter_
                       << ", max=" << limit_.value_or(-1);
  return true;
}

void RetransmissionErrorCounter::Clear() {
  if (counter_ > 0) {
    RTC_DLOG(LS_VERBOSE) << log_prefix_
                         << "recovered from counter=" << counter_;
    counter_ = 0;
  }
}

}  // namespace dcsctp
