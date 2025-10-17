/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 8, 2025.
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
#include "modules/rtp_rtcp/source/rtcp_packet/compound_packet.h"

#include <memory>
#include <utility>

#include "rtc_base/checks.h"

namespace webrtc {
namespace rtcp {

CompoundPacket::CompoundPacket() = default;

CompoundPacket::~CompoundPacket() = default;

void CompoundPacket::Append(std::unique_ptr<RtcpPacket> packet) {
  RTC_CHECK(packet);
  appended_packets_.push_back(std::move(packet));
}

bool CompoundPacket::Create(uint8_t* packet,
                            size_t* index,
                            size_t max_length,
                            PacketReadyCallback callback) const {
  for (const auto& appended : appended_packets_) {
    if (!appended->Create(packet, index, max_length, callback))
      return false;
  }
  return true;
}

size_t CompoundPacket::BlockLength() const {
  size_t block_length = 0;
  for (const auto& appended : appended_packets_) {
    block_length += appended->BlockLength();
  }
  return block_length;
}

}  // namespace rtcp
}  // namespace webrtc
