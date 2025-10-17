/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 5, 2022.
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
#include "modules/rtp_rtcp/include/rtp_rtcp_defines.h"

#include <string.h>

#include <type_traits>

#include "absl/algorithm/container.h"
#include "api/array_view.h"
#include "modules/rtp_rtcp/source/rtp_packet.h"
#include "modules/rtp_rtcp/source/rtp_packet_to_send.h"

namespace webrtc {

namespace {
constexpr size_t kMidRsidMaxSize = 16;

// Check if passed character is a "token-char" from RFC 4566.
// https://datatracker.ietf.org/doc/html/rfc4566#section-9
//    token-char =          %x21 / %x23-27 / %x2A-2B / %x2D-2E / %x30-39
//                         / %x41-5A / %x5E-7E
bool IsTokenChar(char ch) {
  return ch == 0x21 || (ch >= 0x23 && ch <= 0x27) || ch == 0x2a || ch == 0x2b ||
         ch == 0x2d || ch == 0x2e || (ch >= 0x30 && ch <= 0x39) ||
         (ch >= 0x41 && ch <= 0x5a) || (ch >= 0x5e && ch <= 0x7e);
}
}  // namespace

bool IsLegalMidName(absl::string_view name) {
  return (name.size() <= kMidRsidMaxSize && !name.empty() &&
          absl::c_all_of(name, IsTokenChar));
}

bool IsLegalRsidName(absl::string_view name) {
  return (name.size() <= kMidRsidMaxSize && !name.empty() &&
          absl::c_all_of(name, isalnum));
}

StreamDataCounters::StreamDataCounters() = default;

RtpPacketCounter::RtpPacketCounter(const RtpPacket& packet)
    : header_bytes(packet.headers_size()),
      payload_bytes(packet.payload_size()),
      padding_bytes(packet.padding_size()),
      packets(1) {}

RtpPacketCounter::RtpPacketCounter(const RtpPacketToSend& packet_to_send)
    : RtpPacketCounter(static_cast<const RtpPacket&>(packet_to_send)) {
  total_packet_delay =
      packet_to_send.time_in_send_queue().value_or(TimeDelta::Zero());
}

void RtpPacketCounter::AddPacket(const RtpPacket& packet) {
  ++packets;
  header_bytes += packet.headers_size();
  padding_bytes += packet.padding_size();
  payload_bytes += packet.payload_size();
}

void RtpPacketCounter::AddPacket(const RtpPacketToSend& packet_to_send) {
  AddPacket(static_cast<const RtpPacket&>(packet_to_send));
  total_packet_delay +=
      packet_to_send.time_in_send_queue().value_or(TimeDelta::Zero());
}

}  // namespace webrtc
