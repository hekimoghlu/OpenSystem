/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 10, 2021.
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
#include "logging/rtc_event_log/events/rtc_event_generic_packet_received.h"

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/memory/memory.h"

namespace webrtc {

RtcEventGenericPacketReceived::RtcEventGenericPacketReceived(
    int64_t packet_number,
    size_t packet_length)
    : packet_number_(packet_number), packet_length_(packet_length) {}

RtcEventGenericPacketReceived::RtcEventGenericPacketReceived(
    const RtcEventGenericPacketReceived& packet) = default;

RtcEventGenericPacketReceived::~RtcEventGenericPacketReceived() = default;

std::unique_ptr<RtcEventGenericPacketReceived>
RtcEventGenericPacketReceived::Copy() const {
  return absl::WrapUnique(new RtcEventGenericPacketReceived(*this));
}

}  // namespace webrtc
