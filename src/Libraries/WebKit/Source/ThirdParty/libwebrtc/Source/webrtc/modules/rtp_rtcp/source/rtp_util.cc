/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 31, 2022.
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
#include "modules/rtp_rtcp/source/rtp_util.h"

#include <cstddef>
#include <cstdint>

#include "api/array_view.h"
#include "modules/rtp_rtcp/source/byte_io.h"
#include "rtc_base/checks.h"

namespace webrtc {
namespace {

constexpr uint8_t kRtpVersion = 2;
constexpr size_t kMinRtpPacketLen = 12;
constexpr size_t kMinRtcpPacketLen = 4;

bool HasCorrectRtpVersion(rtc::ArrayView<const uint8_t> packet) {
  return packet[0] >> 6 == kRtpVersion;
}

// For additional details, see http://tools.ietf.org/html/rfc5761#section-4
bool PayloadTypeIsReservedForRtcp(uint8_t payload_type) {
  return 64 <= payload_type && payload_type < 96;
}

}  // namespace

bool IsRtpPacket(rtc::ArrayView<const uint8_t> packet) {
  return packet.size() >= kMinRtpPacketLen && HasCorrectRtpVersion(packet) &&
         !PayloadTypeIsReservedForRtcp(packet[1] & 0x7F);
}

bool IsRtcpPacket(rtc::ArrayView<const uint8_t> packet) {
  return packet.size() >= kMinRtcpPacketLen && HasCorrectRtpVersion(packet) &&
         PayloadTypeIsReservedForRtcp(packet[1] & 0x7F);
}

int ParseRtpPayloadType(rtc::ArrayView<const uint8_t> rtp_packet) {
  RTC_DCHECK(IsRtpPacket(rtp_packet));
  return rtp_packet[1] & 0x7F;
}

uint16_t ParseRtpSequenceNumber(rtc::ArrayView<const uint8_t> rtp_packet) {
  RTC_DCHECK(IsRtpPacket(rtp_packet));
  return ByteReader<uint16_t>::ReadBigEndian(rtp_packet.data() + 2);
}

uint32_t ParseRtpSsrc(rtc::ArrayView<const uint8_t> rtp_packet) {
  RTC_DCHECK(IsRtpPacket(rtp_packet));
  return ByteReader<uint32_t>::ReadBigEndian(rtp_packet.data() + 8);
}

}  // namespace webrtc
