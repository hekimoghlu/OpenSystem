/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 1, 2022.
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
#include "rtc_base/network/sent_packet.h"

namespace rtc {

PacketInfo::PacketInfo() = default;
PacketInfo::PacketInfo(const PacketInfo& info) = default;
PacketInfo::~PacketInfo() = default;

SentPacket::SentPacket() = default;
SentPacket::SentPacket(int64_t packet_id, int64_t send_time_ms)
    : packet_id(packet_id), send_time_ms(send_time_ms) {}
SentPacket::SentPacket(int64_t packet_id,
                       int64_t send_time_ms,
                       const rtc::PacketInfo& info)
    : packet_id(packet_id), send_time_ms(send_time_ms), info(info) {}

}  // namespace rtc
