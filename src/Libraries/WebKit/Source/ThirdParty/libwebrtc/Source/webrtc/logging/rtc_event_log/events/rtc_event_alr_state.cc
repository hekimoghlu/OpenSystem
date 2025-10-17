/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 12, 2023.
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
#include "logging/rtc_event_log/events/rtc_event_alr_state.h"

#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "api/rtc_event_log/rtc_event.h"
#include "logging/rtc_event_log/events/rtc_event_definition.h"
#include "logging/rtc_event_log/events/rtc_event_log_parse_status.h"

namespace webrtc {
constexpr RtcEvent::Type RtcEventAlrState::kType;
constexpr RtcEventDefinition<RtcEventAlrState, LoggedAlrStateEvent, bool>
    RtcEventAlrState::definition_;

RtcEventAlrState::RtcEventAlrState(bool in_alr) : in_alr_(in_alr) {}

RtcEventAlrState::RtcEventAlrState(const RtcEventAlrState& other)
    : RtcEvent(other.timestamp_us_), in_alr_(other.in_alr_) {}

RtcEventAlrState::~RtcEventAlrState() = default;

std::unique_ptr<RtcEventAlrState> RtcEventAlrState::Copy() const {
  return absl::WrapUnique<RtcEventAlrState>(new RtcEventAlrState(*this));
}

RtcEventLogParseStatus RtcEventAlrState::Parse(
    absl::string_view s,
    bool batched,
    std::vector<LoggedAlrStateEvent>& output) {
  return RtcEventAlrState::definition_.ParseBatch(s, batched, output);
}

}  // namespace webrtc
