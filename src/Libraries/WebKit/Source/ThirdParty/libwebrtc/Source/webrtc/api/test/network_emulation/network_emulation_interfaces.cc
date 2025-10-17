/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 16, 2024.
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
#include "api/test/network_emulation/network_emulation_interfaces.h"

#include <cstdint>

#include "api/units/data_rate.h"
#include "api/units/timestamp.h"
#include "rtc_base/checks.h"
#include "rtc_base/copy_on_write_buffer.h"
#include "rtc_base/net_helper.h"
#include "rtc_base/net_helpers.h"
#include "rtc_base/socket_address.h"

namespace webrtc {

EmulatedIpPacket::EmulatedIpPacket(const rtc::SocketAddress& from,
                                   const rtc::SocketAddress& to,
                                   rtc::CopyOnWriteBuffer data,
                                   Timestamp arrival_time,
                                   uint16_t application_overhead)
    : from(from),
      to(to),
      data(data),
      headers_size(to.ipaddr().overhead() + application_overhead +
                   cricket::kUdpHeaderSize),
      arrival_time(arrival_time) {
  RTC_DCHECK(to.family() == AF_INET || to.family() == AF_INET6);
}

DataRate EmulatedNetworkOutgoingStats::AverageSendRate() const {
  RTC_DCHECK_GE(packets_sent, 2);
  RTC_DCHECK(first_packet_sent_time.IsFinite());
  RTC_DCHECK(last_packet_sent_time.IsFinite());
  return (bytes_sent - first_sent_packet_size) /
         (last_packet_sent_time - first_packet_sent_time);
}

DataRate EmulatedNetworkIncomingStats::AverageReceiveRate() const {
  RTC_DCHECK_GE(packets_received, 2);
  RTC_DCHECK(first_packet_received_time.IsFinite());
  RTC_DCHECK(last_packet_received_time.IsFinite());
  return (bytes_received - first_received_packet_size) /
         (last_packet_received_time - first_packet_received_time);
}

}  // namespace webrtc
