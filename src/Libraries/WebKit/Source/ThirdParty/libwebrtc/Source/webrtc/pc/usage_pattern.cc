/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 1, 2022.
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
#include "pc/usage_pattern.h"

#include "api/peer_connection_interface.h"
#include "rtc_base/logging.h"
#include "system_wrappers/include/metrics.h"

namespace webrtc {

void UsagePattern::NoteUsageEvent(UsageEvent event) {
  usage_event_accumulator_ |= static_cast<int>(event);
}

void UsagePattern::ReportUsagePattern(PeerConnectionObserver* observer) const {
  RTC_DLOG(LS_INFO) << "Usage signature is " << usage_event_accumulator_;
  RTC_HISTOGRAM_ENUMERATION_SPARSE("WebRTC.PeerConnection.UsagePattern",
                                   usage_event_accumulator_,
                                   static_cast<int>(UsageEvent::MAX_VALUE));
  const int bad_bits =
      static_cast<int>(UsageEvent::SET_LOCAL_DESCRIPTION_SUCCEEDED) |
      static_cast<int>(UsageEvent::CANDIDATE_COLLECTED);
  const int good_bits =
      static_cast<int>(UsageEvent::SET_REMOTE_DESCRIPTION_SUCCEEDED) |
      static_cast<int>(UsageEvent::REMOTE_CANDIDATE_ADDED) |
      static_cast<int>(UsageEvent::ICE_STATE_CONNECTED);
  if ((usage_event_accumulator_ & bad_bits) == bad_bits &&
      (usage_event_accumulator_ & good_bits) == 0) {
    // If called after close(), we can't report, because observer may have
    // been deallocated, and therefore pointer is null. Write to log instead.
    if (observer) {
      observer->OnInterestingUsage(usage_event_accumulator_);
    } else {
      RTC_LOG(LS_INFO) << "Interesting usage signature "
                       << usage_event_accumulator_
                       << " observed after observer shutdown";
    }
  }
}

}  // namespace webrtc
