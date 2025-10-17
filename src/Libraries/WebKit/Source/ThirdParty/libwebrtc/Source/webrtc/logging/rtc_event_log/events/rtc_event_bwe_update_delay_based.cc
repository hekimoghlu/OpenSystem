/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 2, 2023.
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
#include "logging/rtc_event_log/events/rtc_event_bwe_update_delay_based.h"

#include <cstdint>
#include <memory>

#include "absl/memory/memory.h"
#include "api/rtc_event_log/rtc_event.h"
#include "api/transport/bandwidth_usage.h"
#include "logging/rtc_event_log/events/rtc_event_definition.h"

namespace webrtc {

constexpr RtcEventDefinition<RtcEventBweUpdateDelayBased,
                             LoggedBweDelayBasedUpdate,
                             int32_t,
                             BandwidthUsage>
    RtcEventBweUpdateDelayBased::definition_;

RtcEventBweUpdateDelayBased::RtcEventBweUpdateDelayBased(
    int32_t bitrate_bps,
    BandwidthUsage detector_state)
    : bitrate_bps_(bitrate_bps), detector_state_(detector_state) {}

RtcEventBweUpdateDelayBased::RtcEventBweUpdateDelayBased(
    const RtcEventBweUpdateDelayBased& other)
    : RtcEvent(other.timestamp_us_),
      bitrate_bps_(other.bitrate_bps_),
      detector_state_(other.detector_state_) {}

RtcEventBweUpdateDelayBased::~RtcEventBweUpdateDelayBased() = default;

std::unique_ptr<RtcEventBweUpdateDelayBased> RtcEventBweUpdateDelayBased::Copy()
    const {
  return absl::WrapUnique<RtcEventBweUpdateDelayBased>(
      new RtcEventBweUpdateDelayBased(*this));
}

}  // namespace webrtc
