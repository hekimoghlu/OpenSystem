/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 26, 2025.
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
#include "logging/rtc_event_log/events/rtc_event_rtp_packet_incoming.h"

#include <memory>

#include "absl/memory/memory.h"
#include "api/rtc_event_log/rtc_event.h"
#include "modules/rtp_rtcp/source/rtp_packet_received.h"

namespace webrtc {

RtcEventRtpPacketIncoming::RtcEventRtpPacketIncoming(
    const RtpPacketReceived& packet)
    : packet_(packet) {}

RtcEventRtpPacketIncoming::RtcEventRtpPacketIncoming(
    const RtcEventRtpPacketIncoming& other)
    : RtcEvent(other.timestamp_us_), packet_(other.packet_) {}

RtcEventRtpPacketIncoming::~RtcEventRtpPacketIncoming() = default;

std::unique_ptr<RtcEventRtpPacketIncoming> RtcEventRtpPacketIncoming::Copy()
    const {
  return absl::WrapUnique<RtcEventRtpPacketIncoming>(
      new RtcEventRtpPacketIncoming(*this));
}

}  // namespace webrtc
