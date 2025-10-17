/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 7, 2023.
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
#include "logging/rtc_event_log/events/rtc_event_generic_packet_sent.h"

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/memory/memory.h"

namespace webrtc {

RtcEventGenericPacketSent::RtcEventGenericPacketSent(int64_t packet_number,
                                                     size_t overhead_length,
                                                     size_t payload_length,
                                                     size_t padding_length)
    : packet_number_(packet_number),
      overhead_length_(overhead_length),
      payload_length_(payload_length),
      padding_length_(padding_length) {}

RtcEventGenericPacketSent::RtcEventGenericPacketSent(
    const RtcEventGenericPacketSent& packet) = default;

RtcEventGenericPacketSent::~RtcEventGenericPacketSent() = default;

std::unique_ptr<RtcEventGenericPacketSent> RtcEventGenericPacketSent::Copy()
    const {
  return absl::WrapUnique(new RtcEventGenericPacketSent(*this));
}

}  // namespace webrtc
