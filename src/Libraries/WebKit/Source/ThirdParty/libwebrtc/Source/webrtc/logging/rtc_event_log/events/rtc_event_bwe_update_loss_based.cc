/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 27, 2023.
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
#include "logging/rtc_event_log/events/rtc_event_bwe_update_loss_based.h"

#include <cstdint>
#include <memory>

#include "absl/memory/memory.h"
#include "api/rtc_event_log/rtc_event.h"

namespace webrtc {

RtcEventBweUpdateLossBased::RtcEventBweUpdateLossBased(int32_t bitrate_bps,
                                                       uint8_t fraction_loss,
                                                       int32_t total_packets)
    : bitrate_bps_(bitrate_bps),
      fraction_loss_(fraction_loss),
      total_packets_(total_packets) {}

RtcEventBweUpdateLossBased::RtcEventBweUpdateLossBased(
    const RtcEventBweUpdateLossBased& other)
    : RtcEvent(other.timestamp_us_),
      bitrate_bps_(other.bitrate_bps_),
      fraction_loss_(other.fraction_loss_),
      total_packets_(other.total_packets_) {}

RtcEventBweUpdateLossBased::~RtcEventBweUpdateLossBased() = default;

std::unique_ptr<RtcEventBweUpdateLossBased> RtcEventBweUpdateLossBased::Copy()
    const {
  return absl::WrapUnique<RtcEventBweUpdateLossBased>(
      new RtcEventBweUpdateLossBased(*this));
}

}  // namespace webrtc
