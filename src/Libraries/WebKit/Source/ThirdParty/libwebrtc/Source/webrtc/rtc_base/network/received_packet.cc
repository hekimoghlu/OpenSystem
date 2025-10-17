/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 10, 2022.
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
#include "rtc_base/network/received_packet.h"

#include <optional>
#include <utility>

#include "rtc_base/socket_address.h"

namespace rtc {

ReceivedPacket::ReceivedPacket(rtc::ArrayView<const uint8_t> payload,
                               const SocketAddress& source_address,
                               std::optional<webrtc::Timestamp> arrival_time,
                               EcnMarking ecn,
                               DecryptionInfo decryption)
    : payload_(payload),
      arrival_time_(std::move(arrival_time)),
      source_address_(source_address),
      ecn_(ecn),
      decryption_info_(decryption) {}

ReceivedPacket ReceivedPacket::CopyAndSet(
    DecryptionInfo decryption_info) const {
  return ReceivedPacket(payload_, source_address_, arrival_time_, ecn_,
                        decryption_info);
}

// static
ReceivedPacket ReceivedPacket::CreateFromLegacy(
    const uint8_t* data,
    size_t size,
    int64_t packet_time_us,
    const rtc::SocketAddress& source_address) {
  RTC_DCHECK(packet_time_us == -1 || packet_time_us >= 0);
  return ReceivedPacket(rtc::MakeArrayView(data, size), source_address,
                        (packet_time_us >= 0)
                            ? std::optional<webrtc::Timestamp>(
                                  webrtc::Timestamp::Micros(packet_time_us))
                            : std::nullopt);
}

}  // namespace rtc
